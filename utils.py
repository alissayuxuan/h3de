import os
import torch
import numpy as np

def setgpu(gpus):
    if gpus=='all':
        gpus = '0,1,2,3'
    print('using gpu '+gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))


def metric(heatmap, spacing, landmarks):
    N = heatmap.shape[0]
    n_class = heatmap.shape[1]
    total_mre = []
    max_num = 500
    hits = np.zeros((8, n_class))

    for j in range(N):
        cur_mre_group = []
        for i in range(n_class):
            max_count = 0
            group_rate = 0.999
            if np.max(heatmap[j,i])>0:
                while max_count < max_num:
                    h_score_idxs = np.where(
                        heatmap[j, i] >= np.max(heatmap[j, i])*group_rate)
                    group_rate = group_rate - 0.1
                    max_count = len(h_score_idxs[0])
            else:
                h_score_idxs = np.where(
                    heatmap[j, i] >= np.max(heatmap[j, i])*(1+0.5))
            
            h_predict_location = np.array(
                [np.mean(h_score_idxs[0]), np.mean(h_score_idxs[1]), np.mean(h_score_idxs[2])])

            cur_mre = np.linalg.norm(
                np.array(landmarks[j,i] - h_predict_location)*spacing, ord=2)

            if np.mean(landmarks[j, i])>0:
                cur_mre_group.append(cur_mre) 
                hits[4:, i] += 1
                if cur_mre <= 2.0:
                    hits[0, i] += 1
                if cur_mre <= 2.5:
                    hits[1, i] += 1
                if cur_mre <= 3.:
                    hits[2, i] += 1
                if cur_mre <= 4.:
                    hits[3, i] += 1
            else:
                cur_mre_group.append(-1) 
        total_mre.append(np.array(cur_mre_group))

    return total_mre, hits
    
def metric_proj(heatmap, spacing, landmarks, surface_masks=None):
    """
    Extended metric function with optional surface projection of predictions.
    
    Args:
        heatmap: Predicted heatmaps [N, n_class, D, H, W]
        spacing: Voxel spacing for each sample
        landmarks: Ground truth landmarks [N, n_class, 3]
        surface_masks: Optional surface masks for projection [N, D, H, W]
    
    Returns:
        If surface_masks is None:
            total_mre, hits
        If surface_masks is provided:
            total_mre, hits, total_mre_projected, hits_projected
    """
    N = heatmap.shape[0]
    n_class = heatmap.shape[1]
    
    total_mre = []
    total_mre_projected = [] if surface_masks is not None else None
    
    max_num = 500
    hits = np.zeros((8, n_class))
    hits_projected = np.zeros((8, n_class)) if surface_masks is not None else None

    for j in range(N):
        cur_mre_group = []
        cur_mre_projected_group = [] if surface_masks is not None else None
        
        # Get surface for this batch if available
        surface = None
        if surface_masks is not None:
            surface = surface_masks[j]
            if isinstance(surface, np.ndarray):
                surface = torch.from_numpy(surface).float()
        
        for i in range(n_class):
            # ==================== HEATMAP TO PREDICTION ====================
            # Diese Berechnung nur EINMAL machen!
            max_count = 0
            group_rate = 0.999
            if np.max(heatmap[j,i]) > 0:
                while max_count < max_num:
                    h_score_idxs = np.where(
                        heatmap[j, i] >= np.max(heatmap[j, i])*group_rate)
                    group_rate = group_rate - 0.1
                    max_count = len(h_score_idxs[0])
            else:
                h_score_idxs = np.where(
                    heatmap[j, i] >= np.max(heatmap[j, i])*(1+0.5))
            
            h_predict_location = np.array(
                [np.mean(h_score_idxs[0]), np.mean(h_score_idxs[1]), np.mean(h_score_idxs[2])])

            # ==================== ORIGINAL METRIC ====================
            cur_mre = np.linalg.norm(
                np.array(landmarks[j,i] - h_predict_location)*spacing, ord=2)

            if np.mean(landmarks[j, i]) > 0:
                cur_mre_group.append(cur_mre) 
                hits[4:, i] += 1
                if cur_mre <= 2.0:
                    hits[0, i] += 1
                if cur_mre <= 2.5:
                    hits[1, i] += 1
                if cur_mre <= 3.:
                    hits[2, i] += 1
                if cur_mre <= 4.:
                    hits[3, i] += 1
            else:
                cur_mre_group.append(-1)
            
            # ==================== PROJECTED METRIC (wenn surface gegeben) ====================
            if surface is not None:
                from misc import surface_project_coords
                
                # Skip invalid landmarks
                if np.mean(landmarks[j, i]) > 0:
                    # Project prediction to surface
                    h_predict_location_tensor = torch.from_numpy(
                        h_predict_location.reshape(1, 3)
                    ).float()
                    
                    h_predict_location_projected, _ = surface_project_coords(
                        h_predict_location_tensor,
                        surface
                    )
                    h_predict_location_projected = h_predict_location_projected.numpy().flatten()
                    
                    # MRE with projected prediction
                    cur_mre_projected = np.linalg.norm(
                        np.array(landmarks[j,i] - h_predict_location_projected)*spacing, 
                        ord=2
                    )
                    
                    # Hits
                    cur_mre_projected_group.append(cur_mre_projected)
                    hits_projected[4:, i] += 1
                    if cur_mre_projected <= 2.0:
                        hits_projected[0, i] += 1
                    if cur_mre_projected <= 2.5:
                        hits_projected[1, i] += 1
                    if cur_mre_projected <= 3.:
                        hits_projected[2, i] += 1
                    if cur_mre_projected <= 4.:
                        hits_projected[3, i] += 1
                else:
                    cur_mre_projected_group.append(-1)
        
        total_mre.append(np.array(cur_mre_group))
        if surface is not None:
            total_mre_projected.append(np.array(cur_mre_projected_group))
    
    # Return
    if surface_masks is not None:
        return (total_mre, hits, 
                np.array(total_mre_projected), hits_projected)
    else:
        return total_mre, hits

def metricNew(heatmap, spacing, landmarks, output_file='predicted_coordinates.npy'):
    N = heatmap.shape[0]
    n_class = heatmap.shape[1]
    total_mre = []
    max_num = 500
    hits = np.zeros((8, n_class))
    predicted_coordinates = []  # 用于存储预测的坐标

    for j in range(N):
        cur_mre_group = []
        cur_predicted_coords = []  # 存储当前样本的预测坐标
        for i in range(n_class):
            max_count = 0
            group_rate = 0.999
            if np.max(heatmap[j, i]) > 0:
                while max_count < max_num:
                    h_score_idxs = np.where(
                        heatmap[j, i] >= np.max(heatmap[j, i]) * group_rate)
                    group_rate = group_rate - 0.1
                    max_count = len(h_score_idxs[0])
            else:
                h_score_idxs = np.where(
                    heatmap[j, i] >= np.max(heatmap[j, i]) * (1 + 0.5))

            # # 预测的坐标（物理坐标）
            # h_predict_location = np.array(
            #     [np.mean(h_score_idxs[0]), np.mean(h_score_idxs[1]), np.mean(h_score_idxs[2])]) * spacing
            # cur_predicted_coords.append(h_predict_location)  # 添加到当前样本的预测坐标列表

            h_predict_location = np.array(
                [np.mean(h_score_idxs[0]), np.mean(h_score_idxs[1]), np.mean(h_score_idxs[2])])
            # 计算 MRE
            cur_mre = np.linalg.norm(
                np.array(landmarks[j,i] - h_predict_location)*spacing, ord=2)

            if np.mean(landmarks[j, i]) > 0:
                cur_mre_group.append(cur_mre)
                # 添加到当前样本的预测坐标列表
                cur_predicted_coords.append(h_predict_location)

                hits[4:, i] += 1
                if cur_mre <= 2.0:
                    hits[0, i] += 1
                if cur_mre <= 2.5:
                    hits[1, i] += 1
                if cur_mre <= 3.:
                    hits[2, i] += 1
                if cur_mre <= 4.:
                    hits[3, i] += 1
            else:
                cur_mre_group.append(-1)
                # 创建一个与 h_predict_location 大小相同的全-1数组
                h_predict_location_neg = np.ones_like(h_predict_location) * -1
                # 将其添加到 cur_predicted_coords 中
                cur_predicted_coords.append(h_predict_location_neg)
        total_mre.append(np.array(cur_mre_group))
        predicted_coordinates.append(cur_predicted_coords)  # 将当前样本的预测坐标添加到总列表

    # 将预测的坐标保存到 .npy 文件
    predicted_coordinates = np.array(predicted_coordinates)  # 转换为 numpy 数组
    np.save(output_file, predicted_coordinates)  # 保存到文件
    #, predicted_coordinates
    return total_mre, hits


def min_distance_voting(landmarks):
    min_dis = 1000000
    min_landmark = landmarks[0]
    for landmark in landmarks:
        cur_dis = 0
        for sub_landmark in landmarks:
            cur_dis += np.linalg.norm(
            np.array(landmark - sub_landmark), ord=2)
        if cur_dis < min_dis:
            min_dis = cur_dis
            min_landmark = landmark
    return min_landmark


def metric_proposal(proposal_map, spacing, 
                     landmarks,output_file, shrink=4., anchors=[0.5, 1, 1.5, 2], n_class=14):
    # selected number for candidate landmark voting for one landmark
    # can be fine-tuned according to anchor numbers
    select_number = 15 
    predicted_coordinates = []  # 用于存储预测的坐标
    
    batch_size = proposal_map.size(0)
    c = proposal_map.size(1)
    w = proposal_map.size(2)
    h = proposal_map.size(3)
    n_anchor = proposal_map.size(4)
    total_mre = []
    hits = np.zeros((8, n_class))
    
    for j in range(batch_size):
        cur_mre_group = []
        cur_predicted_coords = []  # 存储当前样本的预测坐标
        for idx in range(n_class):
            #################### from proposal map to landmarks #########################
            proposal_map_vector = proposal_map[:,:,:,:,:,3+idx].reshape(-1)
            mask = torch.zeros_like(proposal_map_vector)
            _, cur_idx = torch.topk(
                        proposal_map_vector, select_number)
            mask[cur_idx] = 1
            mask_tensor = mask.reshape((batch_size, c, w, h, n_anchor, -1))
            select_index = np.where(mask_tensor.cpu().numpy()==1)
               
            # get predicted position
            pred_pos = []
            for i in range(len(select_index[0])):
                cur_pos = []
                cur_batch = select_index[0][i] 
                cur_c = select_index[1][i]
                cur_w = select_index[2][i]
                cur_h = select_index[3][i]
                cur_anchor = select_index[4][i]
                cur_predict = torch.tanh(proposal_map[cur_batch, cur_c, cur_w, cur_h, cur_anchor, :3]).cpu().numpy()

                cur_pos.append( (np.array([cur_c, cur_w, cur_h]) + cur_predict*anchors[cur_anchor])*shrink )
                pred_pos.append(cur_pos)
            pred_pos = np.array(pred_pos)
 
            cur_mre = np.linalg.norm(
                (np.array(landmarks[j,idx] - min_distance_voting(pred_pos)))*spacing[j], ord=2)
            if cur_mre <= 2.0:
                hits[0, idx] += 1
            if cur_mre <= 2.5:
                hits[1, idx] += 1
            if cur_mre <= 3.:
                hits[2, idx] += 1
            if cur_mre <= 4.:
                hits[3, idx] += 1
                
            if np.mean(landmarks[j, idx])>0:
                cur_mre_group.append(cur_mre) 
                # 添加到当前样本的预测坐标列表
                cur_predicted_coords.append(pred_pos)
                hits[4:, idx] += 1
            else:
                # if landmark nonexist, do not calculate MRE and SDR, using -1 to indicate it
                cur_mre_group.append(-1) 
                # 创建一个与 h_predict_location 大小相同的全-1数组
                h_predict_location_neg = np.ones_like(pred_pos) * -1
                # 将其添加到 cur_predicted_coords 中
                cur_predicted_coords.append(h_predict_location_neg)
        total_mre.append(np.array(cur_mre_group))
        predicted_coordinates.append(cur_predicted_coords)  # 将当前样本的预测坐标添加到总列表
    # 将预测的坐标保存到 .npy 文件
    predicted_coordinates = np.array(predicted_coordinates)  # 转换为 numpy 数组
    # print(output_file)
    np.save(output_file, predicted_coordinates)  # 保存到文件
    # print(predicted_coordinates)
    return total_mre, hits


def metric_proposal_proj(proposal_map, spacing, 
                     landmarks, output_file, 
                     shrink=4., anchors=[0.5, 1, 1.5, 2], n_class=14,
                     surface_masks=None):  # ← NEU
    select_number = 15 
    predicted_coordinates = []
    
    batch_size = proposal_map.size(0)
    c = proposal_map.size(1)
    w = proposal_map.size(2)
    h = proposal_map.size(3)
    n_anchor = proposal_map.size(4)
    
    total_mre = []
    total_mre_projected = [] if surface_masks is not None else None  # ← NEU
    hits = np.zeros((8, n_class))
    hits_projected = np.zeros((8, n_class)) if surface_masks is not None else None  # ← NEU
    
    for j in range(batch_size):
        cur_mre_group = []
        cur_mre_projected_group = [] if surface_masks is not None else None
        cur_predicted_coords = []
        
        # Get surface for this batch if available
        surface = None
        if surface_masks is not None:
            surface = surface_masks[j]
            if isinstance(surface, np.ndarray):
                surface = torch.from_numpy(surface).float()
        
        for idx in range(n_class):
            #################### from proposal map to landmarks #########################
            # Diese Berechnung nur EINMAL machen! ←←← WICHTIG
            proposal_map_vector = proposal_map[:,:,:,:,:,3+idx].reshape(-1)
            mask = torch.zeros_like(proposal_map_vector)
            _, cur_idx = torch.topk(proposal_map_vector, select_number)
            mask[cur_idx] = 1
            mask_tensor = mask.reshape((batch_size, c, w, h, n_anchor, -1))
            select_index = np.where(mask_tensor.cpu().numpy()==1)
               
            # get predicted position (nur EINMAL berechnen!)
            pred_pos = []
            for i in range(len(select_index[0])):
                cur_pos = []
                cur_batch = select_index[0][i] 
                cur_c = select_index[1][i]
                cur_w = select_index[2][i]
                cur_h = select_index[3][i]
                cur_anchor = select_index[4][i]
                cur_predict = torch.tanh(
                    proposal_map[cur_batch, cur_c, cur_w, cur_h, cur_anchor, :3]
                ).cpu().numpy()

                cur_pos.append(
                    (np.array([cur_c, cur_w, cur_h]) + cur_predict*anchors[cur_anchor])*shrink
                )
                pred_pos.append(cur_pos)
            pred_pos = np.array(pred_pos)
            
            # ==================== ORIGINAL METRIC ====================
            voted_pred = min_distance_voting(pred_pos)
            cur_mre = np.linalg.norm(
                (np.array(landmarks[j,idx]) - voted_pred)*spacing[j], ord=2
            )
            
            if cur_mre <= 2.0:
                hits[0, idx] += 1
            if cur_mre <= 2.5:
                hits[1, idx] += 1
            if cur_mre <= 3.:
                hits[2, idx] += 1
            if cur_mre <= 4.:
                hits[3, idx] += 1
            
            # ==================== PROJECTED METRIC (wenn surface gegeben) ====================
            if surface is not None:
                from misc import surface_project_coords
                
                # Skip invalid landmarks
                if np.mean(landmarks[j, idx]) > 0:
                    # Project proposals to surface
                    pred_pos_tensor = torch.from_numpy(
                        pred_pos.reshape(-1, 3)
                    ).float()
                    
                    pred_pos_projected, _ = surface_project_coords(
                        pred_pos_tensor,
                        surface
                    )
                    pred_pos_projected = pred_pos_projected.numpy().reshape(pred_pos.shape)
                    
                    # Voting on projected
                    voted_pred_projected = min_distance_voting(pred_pos_projected)
                    
                    # MRE
                    cur_mre_projected = np.linalg.norm(
                        (np.array(landmarks[j,idx]) - voted_pred_projected)*spacing[j], 
                        ord=2
                    )
                    
                    # Hits
                    if cur_mre_projected <= 2.0:
                        hits_projected[0, idx] += 1
                    if cur_mre_projected <= 2.5:
                        hits_projected[1, idx] += 1
                    if cur_mre_projected <= 3.:
                        hits_projected[2, idx] += 1
                    if cur_mre_projected <= 4.:
                        hits_projected[3, idx] += 1
                    
                    cur_mre_projected_group.append(cur_mre_projected)
                    hits_projected[4:, idx] += 1
                else:
                    cur_mre_projected_group.append(-1)
            
            # Store coordinates
            if np.mean(landmarks[j, idx]) > 0:
                cur_mre_group.append(cur_mre) 
                cur_predicted_coords.append(pred_pos)
                hits[4:, idx] += 1
            else:
                cur_mre_group.append(-1) 
                h_predict_location_neg = np.ones_like(pred_pos) * -1
                cur_predicted_coords.append(h_predict_location_neg)
        
        total_mre.append(np.array(cur_mre_group))
        if surface is not None:
            total_mre_projected.append(np.array(cur_mre_projected_group))
        predicted_coordinates.append(cur_predicted_coords)
    
    # Save predictions
    predicted_coordinates = np.array(predicted_coordinates, dtype=object)
    np.save(output_file, predicted_coordinates)
    
    # Return
    if surface_masks is not None:
        return (total_mre, hits, 
                np.array(total_mre_projected), hits_projected)
    else:
        return total_mre, hits
                