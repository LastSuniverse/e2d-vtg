"""QVHighlights full-train/official-val training with all annotated windows.

QVHighlights has multiple valid windows per query. Retrieval and quality use
maximum IoU over all of them; jitter samples a target window uniformly. The
Charades training implementation and its single-window protocol are unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import BoundaryDataset, collate_boundary
from .detector import (RetrievalRefinement, decode_intervals, interval_iou,
                               make_training_candidates, select_candidates)
from .train_charades import move


class QVDataset(BoundaryDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        windows = np.asarray(self.rows[index]['relevant_windows'], dtype=np.float32)
        if windows.ndim != 2 or windows.shape[1] != 2 or not len(windows):
            raise ValueError('Expected nonempty QV relevant_windows')
        sample['all_gt_seconds'] = windows
        sample['all_gt'] = windows / (sample['video_len'] * self.clip_len)
        return sample


def collate_qv(samples):
    batch = collate_boundary(samples)
    count = max(len(x['all_gt']) for x in samples)
    for key in ['all_gt', 'all_gt_seconds']:
        batch[key] = torch.zeros(len(samples), count, 2)
    batch['gt_mask'] = torch.zeros(len(samples), count, dtype=torch.bool)
    for i, sample in enumerate(samples):
        n = len(sample['all_gt'])
        batch['gt_mask'][i, :n] = True
        for key in ['all_gt', 'all_gt_seconds']:
            batch[key][i, :n] = torch.from_numpy(sample[key])
    return batch


def max_target_iou(pred, gt, mask):
    values = interval_iou(pred[:, :, None, :], gt[:, None, :, :])
    return values.masked_fill(~mask[:, None, :], -1).max(-1).values


def qv_loss(out, gt, gt_mask):
    logits, valid = out['relation_logits'].float(), out['valid']
    b, t, _ = logits.shape
    lengths = valid.any(-1).sum(-1).clamp_min(1).float()
    start = torch.arange(t, device=logits.device).float()[None, :] / lengths[:, None]
    end = (torch.arange(t, device=logits.device).float()[None, :] + 1) / lengths[:, None]
    grid = torch.stack([start[:, :, None].expand(b,t,t), end[:, None, :].expand(b,t,t)], -1)
    iou = max_target_iou(grid.reshape(b,t*t,2), gt, gt_mask).reshape(b,t,t)
    target = torch.softmax((iou/.08).masked_fill(~valid, -1e4).flatten(1), -1)
    retrieval = -(target * logits.flatten(1).log_softmax(-1)).sum(-1).mean()
    candidate_iou = max_target_iou(out['spans'].float(), gt, gt_mask)
    quality = F.binary_cross_entropy_with_logits(out['quality'].float(), candidate_iou.detach())
    refinement = 1-candidate_iou.mean()
    return {'loss': retrieval+.5*quality+.5*refinement, 'retrieval': retrieval,
            'quality': quality, 'refinement': refinement}


@torch.inference_mode()
def evaluate(model, loader, device, k):
    model.eval()
    values, predictions = [], []
    for raw in loader:
        batch = move(raw, device)
        base = model(batch['video'], batch['text'], batch['video_mask'], batch['text_mask'])
        candidates = select_candidates(base['relation_logits'], base['valid'], k)
        refined = model.refine_candidates(base['features'], base['query'], base['relation_logits'],
                                          candidates, batch['video_mask'].sum(-1))
        spans, _ = decode_intervals(refined, batch['durations'], topk=1)
        values.append(max_target_iou(spans, batch['all_gt_seconds'], batch['gt_mask'])[:,0].cpu())
        for i, qid in enumerate(batch['qids']):
            s,e = spans[i,0].tolist()
            predictions.append({'qid': qid, 'pred_relevant_windows': [[s,e,1.0]]})
    v = torch.cat(values)
    metrics = {f'R1@{t}': (v>=t).float().mean().item() for t in [.3,.5,.7]}
    return {**metrics, 'mIoU': v.mean().item()}, predictions


def main():
    p = argparse.ArgumentParser()
    for key in ['train-jsonl','eval-jsonl','train-video-cache','eval-video-cache',
                'train-query-cache','eval-query-cache','output-dir']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=2022)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--expected-train', type=int, default=7218)
    p.add_argument('--expected-eval', type=int, default=1550)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(a.seed)
    sets = []
    for split in ['train','eval']:
        ds = QVDataset(getattr(a,split+'_jsonl'), getattr(a,split+'_video_cache'),
                       getattr(a,split+'_query_cache'), max_video_len=150, clip_len=1)
        if len(ds) != getattr(a,'expected_'+split) or ds.missing_video or ds.missing_query:
            raise RuntimeError(f'Incomplete official {split}: {len(ds)} video_missing={ds.missing_video} query_missing={ds.missing_query}')
        sets.append(ds)
    kw = dict(batch_size=a.batch_size,num_workers=a.workers,collate_fn=collate_qv,
              persistent_workers=a.workers>0,pin_memory=True)
    train = DataLoader(sets[0],shuffle=True,**kw)
    val = DataLoader(sets[1],shuffle=False,**kw)
    model = RetrievalRefinement(2050,2048,192,64,max_len=150).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs*len(train),eta_min=a.lr*.05)
    scaler = torch.amp.GradScaler('cuda',enabled=a.amp)
    dest = Path(a.output_dir);dest.mkdir(parents=True,exist_ok=False)
    (dest/'config.json').write_text(json.dumps(vars(a),indent=2))
    best = -1.
    print('QV_TRAIN_START',len(sets[0]),len(sets[1]),flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();started=time.time();total=0.
        for step,raw in enumerate(train,1):
            batch = move(raw,device)
            gt_index = torch.floor(torch.rand(len(batch['qids']),device=device)*batch['gt_mask'].sum(-1)).long()
            target = batch['all_gt'][torch.arange(len(gt_index),device=device),gt_index]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=a.amp):
                base = model(batch['video'],batch['text'],batch['video_mask'],batch['text_mask'])
                candidates = make_training_candidates(base['relation_logits'].detach(),base['valid'],target,48,12)
                base.update(model.refine_candidates(base['features'],base['query'],base['relation_logits'],candidates,batch['video_mask'].sum(-1)))
                losses = qv_loss(base,batch['all_gt'],batch['gt_mask'])
            if not torch.isfinite(losses['loss']):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}, step {step}')
            scaler.scale(losses['loss']).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if not scaler.is_enabled() or scaler.get_scale()>=before:scheduler.step()
            total+=float(losses['loss'].detach())
            if step%50==0:print('TRAIN',epoch,step,len(train),'loss',total/step,flush=True)
        metrics,preds = evaluate(model,val,device,16)
        row={'epoch':epoch,'train_loss':total/len(train),**metrics,'seconds':time.time()-started,
             'protocol':f'QVHighlights train {len(train)}/{a.expected_train} available queries; official val {len(val)}/{a.expected_eval} each epoch; max IoU over all GT windows',
             'train_queries_used':len(train),'train_queries_expected':a.expected_train,
             'partial_train':len(train) != a.expected_train,
             'seed':a.seed}
        print(json.dumps(row),flush=True)
        with (dest/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        config={**vars(a),'video_dim':2050,'text_dim':2048,'dim':192,'rank':64,'max_video_len':150,'max_len':150,'clip_len':1}
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
               'epoch':epoch,'metrics':row,'config':config}
        torch.save(state,dest/'last.ckpt')
        if metrics['R1@0.5']>best:
            best=metrics['R1@0.5'];torch.save(state,dest/'model_best.ckpt')
            (dest/'best_predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in preds))
            (dest/'best.json').write_text(json.dumps(row,indent=2))


if __name__ == '__main__':
    main()
