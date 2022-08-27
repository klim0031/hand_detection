import torch

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ^ B / A || B = A ^ B / (area(A) + area(B) - A ^ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    #conf = labels[best_truth_idx]         # don't nedd add 1, because indexed from background
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior

def match_v2(threshold, gt_boxes, truths, priors, variances, labels, loc_t, conf_t, idx):
    # jaccard index
    overlaps = jaccard(
        gt_boxes,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    #conf = labels[best_truth_idx]         # don't nedd add 1, because indexed from background
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode_v2(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode_v2(matched, priors, variances):

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:4])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])

    # match wh / prior wh
    g_wh = (matched[:, 2:4] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    #
    g_wxy = torch.log( matched[:, 4:6]/ priors[:, 2:] + 0.1 ) / variances[1]
    g_hxy = torch.log( matched[:, 6:8]/ priors[:, 2:] + 0.1 ) / variances[1]

    #
    wrist_vec = (matched[:, 8:10] - priors[:, :2])/ (variances[0] * priors[:, 2:])

    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh, g_wxy, g_hxy, wrist_vec], 1)  # [num_priors,10]

def decode_v2(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:4] * variances[1]),
        torch.clamp(priors[:, 2:] * (torch.exp(loc[:, 4:6] * variances[1]) - 0.1), min=0),
        torch.clamp(priors[:, 2:] * (torch.exp(loc[:, 6:8] * variances[1]) - 0.1), min=0),
        priors[:, :2] + loc[:, 8:10] * variances[0] * priors[:, 2:]), 1)

    boxes[:, :2] -= boxes[:, 2:4] / 2
    boxes[:, 2:4] += boxes[:, :2]
    return boxes

def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# # Original author: Francisco Massa:
# # https://github.com/fmassa/object-detection.torch
# # Ported to PyTorch by Max deGroot (02/01/2017)
def nms_old(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
#
# def nms_new(boxes, scores, overlap=0.5, top_k=200):
#     """Apply non-maximum suppression at test time to avoid detecting too many
#     overlapping bounding boxes for a given object.
#     Args:
#         boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
#         scores: (tensor) The class predscores for the img, Shape:[num_priors].
#         overlap: (float) The overlap thresh for suppressing unnecessary boxes.
#         top_k: (int) The Maximum number of box preds to consider.
#     Return:
#         The indices of the kept boxes with respect to num_priors.
#     """
#     # print("nms_old:.>>>>>>>>>>>>>>>>>")
#     # nms_old(boxes, scores, overlap, top_k)
#     # print("nms_new:.>>>>>>>>>>>>>>>>>")
#
#     dets = torch.cat((boxes, scores), dim=1)
#     dets = dets.type(torch.FloatTensor)
#     if boxes.numel() == 0:
#         return keep
#
#
#     scores = dets[:,4]
#     idx = scores.sort(0, descending=True)[1]  # sort in ascending order
#
#
#     idx = idx[0:top_k]
#     dets = dets[idx].contiguous()
#
#     scores2 = dets[:,4]
#     idx2 = scores2.sort(0, descending=True)[1]  # sort in ascending order
#
#
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     areas = torch.mul(x2 - x1, y2 - y1)
#
#     # I = I[v >= 0.01]
#     #dets = dets[idx].contiguous()
#     keep = torch.LongTensor(dets.size(0))
#     num_out = torch.LongTensor(1)
#     nms.cpu_nms(keep, num_out, dets, idx2, areas, overlap)
#
#
#     keep = idx[keep[:num_out[0]]].cuda().contiguous()
#
#     return keep,num_out[0]
#
#
# def nms_new_gpu(boxes, scores, overlap=0.5, top_k=200):
#     """Apply non-maximum suppression at test time to avoid detecting too many
#     overlapping bounding boxes for a given object.
#     Args:
#         boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
#         scores: (tensor) The class predscores for the img, Shape:[num_priors].
#         overlap: (float) The overlap thresh for suppressing unnecessary boxes.
#         top_k: (int) The Maximum number of box preds to consider.
#     Return:
#         The indices of the kept boxes with respect to num_priors.
#     """
#     # print("nms_old:.>>>>>>>>>>>>>>>>>")
#     # nms_old(boxes, scores, overlap, top_k)
#     # print("nms_new:.>>>>>>>>>>>>>>>>>")
#
#     dets = torch.cat((boxes, scores), dim=1)
#     if boxes.numel() == 0:
#         return keep
#
#     idx = scores.sort(0, descending=True)[1]  # sort in ascending order
#     # I = I[v >= 0.01]
#     dets = dets[idx[0:top_k]].contiguous()
#     keep = torch.LongTensor(dets.size(0))
#     num_out = torch.LongTensor(1)
#     nms.gpu_nms(keep, num_out, dets, overlap)
#
#     #print(keep.size())
#     #print(num_out.size())
#     #print(idx[keep[:num_out[0]].cuda()].contiguous().size())
#
#     return idx[keep[:num_out[0]].cuda()].contiguous().view(-1),num_out[0]
