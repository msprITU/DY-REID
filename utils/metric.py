import torch


def cosine(query, gallery):
    query = torch.from_numpy(query)
    gallery = torch.from_numpy(gallery)

    m, n = query.size(0), gallery.size(0)
    dist = 1 - torch.mm(query, gallery.t()) / ((torch.norm(query, 2, dim=1, keepdim=True).expand(m, n)
                                                * torch.norm(gallery, 2, dim=1, keepdim=True).expand(n, m).t()))
    return dist.numpy()


def euclidean(query, gallery):
    query = torch.from_numpy(query)
    gallery = torch.from_numpy(gallery)

    m, n = query.size(0), gallery.size(0)
    dist = torch.pow(query, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(gallery, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, query, gallery.t())
    return dist.numpy()


def mask_distance(query, gallery):
    query_gf = torch.from_numpy(query['global_feature']).cuda()
    gallery_gf = torch.from_numpy(gallery['global_feature']).cuda()
    query_lf = torch.from_numpy(query['local_feature']).cuda()
    gallery_lf = torch.from_numpy(gallery['local_feature']).cuda()
    qm = torch.from_numpy(query['part_label']).cuda()
    gm = torch.from_numpy(gallery['part_label']).cuda()
    #######Calculate the distance of pose-guided global features

    global_dist = (1 - torch.mm(query_gf, gallery_gf.t())) / 2
    ########Calculate the distance of partial features
    qm = qm.unsqueeze(dim=1)
    gm = gm.unsqueeze(dim=0)
    overlap = (qm * gm).float()

    local_dists = []
    for i in range(query_lf.size()[0]):
        local_dist_i = (1 - (query_lf[i:i + 1] * gallery_lf).sum(-1)) / 2
        local_dist_i = torch.unsqueeze(local_dist_i, dim=0)
        local_dists.append(local_dist_i)
    local_dists = torch.cat(local_dists, dim=0)
    local_dist = (local_dists * overlap).sum(-1)
    dist = (local_dist + global_dist) / (overlap.sum(-1) + 1)
    dist = dist.cpu().numpy()

    return dist

def part_distance(query, gallery):
    """"""
    qgf, qlf, qlp = query['global_features'], query['local_features'], query['local_parts'][:, :, 1]
    ggf, glf, glp = gallery['global_features'], gallery['local_features'], gallery['local_parts'][:, :, 1]

    sgf = (1. - torch.mm(qgf, ggf.t())) / 2

    qlp, glp = qlp.unsqueeze(1), glp.unsqueeze(0)
    overlap = qlp * glp

    slf = (1. - torch.matmul(qlf.permute(1, 0, 2), glf.permute(1, 2, 0))) / 2
    slf = slf.permute(1, 2, 0) * overlap

    dist = (slf.sum(-1) + sgf) / (overlap.sum(-1) + 1)
    return dist.data.cpu().numpy()