# import torch
# from torch import nn
# import torch.nn.functional as F

import paddle 
import paddle.nn.functional as F
from models.backbone import build_backbone
from models.matcher import build_matcher
from models.transformer import build_hoi_ts
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, is_dist_avail_and_initialized)


class DETRHOI(paddle.nn.Layer):

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries,
                 aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = paddle.nn.Embedding(num_queries, hidden_dim)
        #########################################################
        self.obj_class_embed = paddle.nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed =paddle.nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        #########################################################
        self.input_proj = paddle.nn.Conv2D(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        ##########################################################
        self.query_embed_sp = MLP(12, hidden_dim, hidden_dim, 2)
        self.query_embed_obj = MLP(512, hidden_dim, hidden_dim, 2)

    def forward(self, samples: NestedTensor, gt_items=None):
        gt_q_mask = None
        if self.aux_loss and gt_items is not None:
            max_len = 0
            bs = len(gt_items)
            for item in gt_items:
                max_len = max(max_len, len(item))
            gt_obj_vec = paddle.zeros([max_len, bs, 512]).type_as(gt_items[0])
            gt_sp_vec = paddle.zeros([max_len, bs, 12]).type_as(gt_items[0])
            gt_q_mask = paddle.ones([bs, max_len]).type_as(gt_items[0]).type(paddle.bool)
            #############################################################################
            for i in range(len(gt_items)):
                item = gt_items[i]
                if len(item) == 0:
                    gt_q_mask[i, :] = False
                    continue
                gt_obj_vec[0:len(item), i, :] = item[:, 0:512]
                gt_sp_vec[0:len(item), i, :] = item[:, 512:512+12]
                gt_q_mask[i, 0:len(item)] = False
            # todo
            t_sp = paddle.tanh(self.query_embed_sp(gt_sp_vec))
            t_obj_emb = self.query_embed_obj(gt_obj_vec)
        ##########################################################
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        ##################################################################################################
        if self.aux_loss and gt_items is not None:
            att, att_gt, hs, hs_gt = self.transformer(self.input_proj(src), mask, self.query_embed.weight,
                                                      query_embed_q=t_sp, query_embed_e=t_obj_emb,
                                                      query_embed2_mask=gt_q_mask,
                                                      pos_embed=pos[-1])

            outputs_obj_class = self.obj_class_embed(hs)
            outputs_verb_class = self.verb_class_embed(hs)
            outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
            outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()

            outputs_gt_obj_class = self.obj_class_embed(hs_gt)
            outputs_gt_verb_class = self.verb_class_embed(hs_gt)
            outputs_gt_sub_coord = self.sub_bbox_embed(hs_gt).sigmoid()
            outputs_gt_obj_coord = self.obj_bbox_embed(hs_gt).sigmoid()

            out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                   'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
                   'gt_obj_logits': outputs_gt_obj_class[-1], 'gt_verb_logits': outputs_gt_verb_class[-1],
                   'gt_sub_boxes': outputs_gt_sub_coord[-1], 'gt_obj_boxes': outputs_gt_obj_coord[-1],
                   'hs': hs[-1], 'hs_gt': hs_gt[-1],
                   'att': att[-1], 'att_gt': att_gt[-1]
                   }
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord,
                                                    outputs_gt_obj_class, outputs_gt_verb_class,
                                                    outputs_gt_sub_coord, outputs_gt_obj_coord,
                                                    hs, hs_gt, att, att_gt)
        else:
            hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight,
                                  pos_embed=pos[-1])
            outputs_obj_class = self.obj_class_embed(hs)
            outputs_verb_class = self.verb_class_embed(hs)
            outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
            outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()

            out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                   'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
                   }

        return out

    # @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord,
                      outputs_gt_obj_class, outputs_gt_verb_class, outputs_gt_sub_coord, outputs_gt_obj_coord,
                      hs, hs_gt, att, att_gt):
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d,
                 'gt_obj_logits': a1, 'gt_verb_logits': b1, 'gt_sub_boxes': c1, 'gt_obj_boxes': d1,
                 'hs': h1, 'hs_gt': h2, 'att': att1, 'att_gt': att2}
                for a, b, c, d, a1, b1, c1, d1, h1, h2, att1, att2 in
                zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                    outputs_sub_coord[:-1], outputs_obj_coord[:-1],
                    outputs_gt_obj_class[:-1], outputs_gt_verb_class[:-1],
                    outputs_gt_sub_coord[:-1], outputs_gt_obj_coord[:-1],
                    hs[:-1], hs_gt[:-1],
                    att[:-1], att_gt[:-1]
                    )]


class MLP(paddle.nn.Layer):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers =  paddle.nn.LayerList(paddle.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(paddle.nn.Layer):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses,
                 verb_loss_type):
        super().__init__()

        assert verb_loss_type == 'bce' or verb_loss_type == 'focal'

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = paddle.ones(shape=[self.num_obj_classes + 1])
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.verb_loss_type = verb_loss_type

    def loss_obj_labels(self, outputs, targets, indices, indices_gt, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = paddle.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = paddle.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=paddle.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_gt_obj_labels(self, outputs, targets, indices, indices_gt, num_interactions, log=True):
        src_logits = outputs['gt_obj_logits']
        target_classes_o = paddle.cat([t['obj_labels'] for t in targets])
        src_logits = src_logits[indices_gt]
        if len(target_classes_o) == 0:
            loss_obj_ce = src_logits.sum()
        else:
            loss_obj_ce = F.cross_entropy(src_logits, target_classes_o, self.empty_weight)
        losses = {'loss_gt_obj_ce': loss_obj_ce}
        return losses

    @paddle.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, indices_gt, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = paddle.to_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, indices_gt, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = paddle.concat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = paddle.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        if self.verb_loss_type == 'bce':
            loss_verb_ce = paddle.nn.BCEWithLogitsLoss()(src_logits, target_classes)
        elif self.verb_loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._neg_loss(src_logits, target_classes)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_gt_verb_labels(self, outputs, targets, indices, indices_gt, num_interactions):
        src_logits = outputs['gt_verb_logits']
        target_classes_o = paddle.concat([t['verb_labels'] for t in targets])
        src_logits = src_logits[indices_gt]

        if self.verb_loss_type == 'bce':
            loss_verb_ce = paddle.nn.BCEWithLogitsLoss()(src_logits, target_classes_o)
        elif self.verb_loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._neg_loss(src_logits, target_classes_o)

        losses = {'loss_gt_verb_ce': loss_verb_ce}
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, indices_gt, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = paddle.concat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = paddle.concat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - paddle.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - paddle.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def loss_gt_sub_obj_boxes(self, outputs, targets, indices, indices_gt, num_interactions):
        src_sub_boxes = outputs['gt_sub_boxes'][indices_gt]
        src_obj_boxes = outputs['gt_obj_boxes'][indices_gt]
        target_sub_boxes = paddle.concat([t['sub_boxes'] for t in targets], dim=0)
        target_obj_boxes = paddle.concat([t['obj_boxes'] for t in targets], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_gt_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_gt_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_gt_sub_giou'] = src_sub_boxes.sum()
            losses['loss_gt_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_gt_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_gt_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - paddle.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - paddle.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_gt_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_gt_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def loss_match_cosine(self, outputs, targets, indices, indices_gt, num_interactions):
        hs = outputs['hs']
        hs_gt = outputs['hs_gt']

        idx = self._get_src_permutation_idx(indices)
        target_hs = hs[idx]
        target_hs_gt = paddle.concat([t[i] for t, (_, i) in zip(hs_gt, indices)], dim=0)

        if len(target_hs) > 0:
            loss_match = paddle.nn.CosineEmbeddingLoss()(target_hs, target_hs_gt,
                                                  paddle.ones([target_hs.size()[0]]).to(target_hs_gt.device))
        else:
            loss_match = paddle.zeros([]).type_as(target_hs)
        losses = {'loss_match': loss_match}
        return losses

    def loss_match_kl(self, outputs, targets, indices, indices_gt, num_interactions):
        hs = outputs['att']
        hs_gt = outputs['att_gt']
     
        idx = self._get_src_permutation_idx(indices)
        target_hs = hs[idx]
        target_hs_gt = paddle.concat([t[i] for t, (_, i) in zip(hs_gt, indices)], dim=0)

        if len(target_hs) > 0:
            loss_match = 10 * paddle.nn.KLDivLoss()((target_hs + 1e-6).log(), target_hs_gt) / 3
        else:
            loss_match = paddle.zeros([]).type_as(target_hs)
        losses = {'loss_match_att': loss_match}
        return losses

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = paddle.pow(1 - gt, 4)

        loss = 0

        pos_loss = paddle.log(pred) * paddle.pow(1 - pred, 2) * pos_inds
        neg_loss = paddle.log(1 - pred) * paddle.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = paddle.concat([paddle.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = paddle.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = paddle.concat([paddle.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = paddle.concat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, indices_gt, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes,

            'gt_obj_labels': self.loss_gt_obj_labels,
            'gt_verb_labels': self.loss_gt_verb_labels,
            'gt_sub_obj_boxes': self.loss_gt_sub_obj_boxes,

            'match': self.loss_match_cosine,
            'match_att': self.loss_match_kl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, indices_gt, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = paddle.to_tensor([num_interactions], dtype=paddle.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            paddle.distributed.all_reduce(num_interactions)
        num_interactions = paddle.clip(num_interactions / get_world_size(), min=1).item()

        batch_idx = []
        batch_idx1 = []
        for i, t in enumerate(targets):
            batch_idx.extend([i] * len(t['sub_boxes']))
            batch_idx1.extend(paddle.arange(0, len(t['sub_boxes'])))
        gt_idx = []
        gt_idx.append(batch_idx)
        gt_idx.append(batch_idx1)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, gt_idx, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}

                    if loss == "gt_obj_labels" and 'gt_obj_logits' not in aux_outputs: continue
                    if loss == "gt_verb_labels" and 'gt_verb_logits' not in aux_outputs: continue
                    if loss == "gt_sub_obj_boxes" and 'gt_sub_boxes' not in aux_outputs: continue
                    # todo best for only last match
                    if loss == "match" or (loss == "match_att" and i < 3):
                        continue

                    if loss == 'obj_labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, gt_idx, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcessHOI(paddle.nn.Layer):

    def __init__(self, subject_category_id):
        super().__init__()
        self.subject_category_id = subject_category_id

    @paddle.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = paddle.nn.Softmax()(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = paddle.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = paddle.full_like(ol, self.subject_category_id)
            l = paddle.concat((sl, ol))
            b = paddle.concat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})
            vs = vs * os.unsqueeze(1)
            ids = paddle.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})
        return results


class PostProcessVCOCO(paddle.nn.Layer):

    def __init__(self, num_queries, subject_category_id, correct_mat):
        import numpy as np
        super().__init__()
        self.max_hois = 100

        self.num_queries = num_queries
        self.subject_category_id = subject_category_id

        correct_mat = np.concatenate((correct_mat, np.ones((correct_mat.shape[0], 1))), axis=1)
        self.register_buffer('correct_mat', paddle.to_tensor(correct_mat))

    @paddle.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = paddle.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = paddle.full_like(ol, self.subject_category_id)
            l = paddle.concat((sl, ol))
            b = paddle.concat((sb, ob))
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in
                      zip(b.to('cpu').numpy(), l.to('cpu').numpy())]

            hoi_scores = vs * os.unsqueeze(1)

            verb_labels = paddle.arange(hoi_scores.shape[1], device=self.correct_mat.device).view(1, -1).expand(
                hoi_scores.shape[0], -1)
            object_labels = ol.view(-1, 1).expand(-1, hoi_scores.shape[1])
            masks = self.correct_mat[verb_labels.reshape(-1), object_labels.reshape(-1)].view(hoi_scores.shape)
            hoi_scores *= masks

            ids = paddle.arange(b.shape[0])

            hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                    subject_id, object_id, category_id, score in zip(ids[:ids.shape[0] // 2].to('cpu').numpy(),
                                                                     ids[ids.shape[0] // 2:].to('cpu').numpy(),
                                                                     verb_labels.to('cpu').numpy(),
                                                                     hoi_scores.to('cpu').numpy())]

            results.append({
                'predictions': bboxes,
                'hoi_prediction': hois
            })

        return results


def build(args):
    # device = torch.device(args.device)
    device = paddle.device.get_device()
    backbone = build_backbone(args)
    matcher = build_matcher(args)

    transformer = build_hoi_ts(args)

    model = DETRHOI(
        backbone,
        transformer,
        num_obj_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef

    weight_dict['loss_gt_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_gt_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_gt_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_gt_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_gt_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_gt_obj_giou'] = args.giou_loss_coef
    weight_dict['loss_match'] = 1
    weight_dict['loss_match_att'] = 1

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality',
              'gt_obj_labels', 'gt_verb_labels', 'gt_sub_obj_boxes', 'match', 'match_att']
    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                verb_loss_type=args.verb_loss_type)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOI(args.subject_category_id)}

    return model, criterion, postprocessors
