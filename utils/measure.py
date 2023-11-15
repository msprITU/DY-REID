import numpy as np


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    #with open('/content/drive/MyDrive/PersonReID-YouReID/matrisler/distmat_baseline_dyn_occduke_10epoch.npy', 'wb') as file_distance:
    #    np.save(file_distance, distmat)

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1) #kucukten buyuge siralayip indisleri donduruyor (satir bazli siralama yapiyor)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    #matches: sutun sutun karsilastiriyo (elementwise), ayni id ise 1, degilse 0
    #g_pids[indices]: satir bazinda en benzeyenden en benzemeyene sirali idler
    #indices ve matches numpy ndarray
    
    #----------------------------------EEA
    """
    with open('/content/drive/MyDrive/PersonReID-YouReID/cacenet_duke_matrices/matches_cacenet_duke_static.npy', 'wb') as file_matches:
        np.save(file_matches, matches)

    with open('/content/drive/MyDrive/PersonReID-YouReID/cacenet_duke_matrices/indices_cacenet_duke_static.npy', 'wb') as file_indices:
        np.save(file_indices, indices)

    with open('/content/drive/MyDrive/PersonReID-YouReID/cacenet_duke_matrices/distmat_cacenet_duke_static.npy', 'wb') as file_distmat:
        np.save(file_distmat, distmat)
    """

    #for i in range(10): #----eea
    #  print(indices[3238, i])

    #print(indices[3067,:])
    #print(indices[3068,:])

    #print(g_pids[indices])
    #print(q_pids[:, np.newaxis])
    #print(matches)

    #----------------------------------EEA
    # compute cmc curve for each query
    all_cmc = []
    all_ap = []
    num_valid_q = 0.  # number of valid query
    all_raw_cmc = np.zeros((num_q, num_g)) #----EEA

    for q_idx in range(num_q): #num_q : query sample sayisi
        # get query pid and camid
        q_pid = q_pids[q_idx] #1den 1501'e (markette)
        q_camid = q_camids[q_idx] #kamera idleri (sirali degil)

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches

        #------EEA
        all_raw_cmc[q_idx, :raw_cmc.shape[0]] = raw_cmc

        #if q_idx == 522: #----eea
        #    print(raw_cmc.sum())
        #    print(q_pid)
        #    for i in range(10):
        #        print(raw_cmc[i])

        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        ap = tmp_cmc.sum() / num_rel

        #----------------------------EEA sinif bazli AP raporlama
        #with open('/content/drive/MyDrive/PersonReID-YouReID/cacenet_duke_matrices/AP_file_cacenet_duke_static.txt', 'a') as ap_file:
        #    ap_file.write('{}\t{}\n'.format(q_pid, ap))
        #----------------------------EEA
        all_ap.append(ap)

    #-------EEA raw cmc kaydetme yeni metrik icin:
    #all_raw_cmc = np.array(all_raw_cmc)
    #print(all_raw_cmc)
    #print(all_raw_cmc.shape)
    #with open('/content/drive/MyDrive/PersonReID-YouReID/cacenet_duke_matrices/all_raw_cmc_cacenet_duke_static.npy', 'wb') as file_rawcmc:
    #    np.save(file_rawcmc, all_raw_cmc)
    #----------------------------EEA

    #print(len(all_ap)) #query sayisi kadar 
    if num_valid_q < 0:
        mAP = 0
        all_cmc = [0 for i in range(100)]
        return all_cmc, mAP

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_ap)

    return all_cmc, mAP


def evaluate_py(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)


def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluates CMC rank.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    return evaluate_py(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
