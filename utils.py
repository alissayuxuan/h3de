import os
import torch
import numpy as np
from TPTBox import POI

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
    total_mse = []
    total_mre_projected = [] if surface_masks is not None else None
    total_mse_projected = [] if surface_masks is not None else None

    max_num = 500
    hits = np.zeros((8, n_class))
    hits_projected = np.zeros((8, n_class)) if surface_masks is not None else None

    for j in range(N):
        cur_mre_group = []
        cur_mse_group = []
        cur_mre_projected_group = [] if surface_masks is not None else None
        cur_mse_projected_group = [] if surface_masks is not None else None
        
        # Get surface for this batch if available
        surface = None
        if surface_masks is not None:
            surface = surface_masks[j]
            if isinstance(surface, np.ndarray):
                surface = torch.from_numpy(surface).float()
        
        for i in range(n_class):
            # ==================== HEATMAP TO PREDICTION ====================
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
            cur_mse = cur_mre**2

            if np.mean(landmarks[j, i]) > 0:
                cur_mre_group.append(cur_mre)
                cur_mse_group.append(cur_mse) 
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
                cur_mse_group.append(-1)
            
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
                    cur_mse_projected = cur_mre_projected**2
                    
                    # Hits
                    cur_mre_projected_group.append(cur_mre_projected)
                    cur_mse_projected_group.append(cur_mse_projected)

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
                    cur_mse_projected_group.append(-1)
        
        total_mre.append(np.array(cur_mre_group))
        total_mse.append(np.array(cur_mse_group))
        if surface is not None:
            total_mre_projected.append(np.array(cur_mre_projected_group))
            total_mse_projected.append(np.array(cur_mse_projected_group))   
    
    # Return
    #if surface_masks is not None:
    #    return (total_mre, hits, 
    #            np.array(total_mre_projected), hits_projected)
    #else:
    #    return total_mre, hits
    results = {
        'mre': (total_mre, hits),
        'mse': (total_mse, hits)
    }
    if surface is not None:
        results['mre_projected'] = (total_mre_projected, hits_projected)
        results['mse_projected'] = (total_mse_projected, hits_projected)

    return results

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
                    surface_masks=None,
                    poi_metadata=None,
                    poi_save_dir=None,
                    crop_rate=1,
                    crop_begin=0
                    ): 
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
        
        voted_predictions = [] # collect predictions for this sample

        for idx in range(n_class):
            #################### from proposal map to landmarks #########################
            proposal_map_vector = proposal_map[:,:,:,:,:,3+idx].reshape(-1)
            mask = torch.zeros_like(proposal_map_vector)
            _, cur_idx = torch.topk(proposal_map_vector, select_number)
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

            if voted_pred.ndim > 1:
                voted_pred = voted_pred.squeeze()

            voted_predictions.append((idx, voted_pred))  # save prediction for this landmark

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
        
        # ======== Save POI ===========

        if poi_metadata is not None:
            _save_poi_files_for_sample(
                batch_idx=j,
                voted_predictions=voted_predictions,
                gt_landmarks=landmarks[j],
                poi_metadata=poi_metadata,
                save_dir=poi_save_dir,
                crop_rate=crop_rate,
                crop_begin=crop_begin
            )
            
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


def _save_poi_files_for_sample(batch_idx, voted_predictions, gt_landmarks, 
                                poi_metadata, save_dir, crop_rate, crop_begin):
    """
    Hilfsfunktion zum Speichern von GT und Prediction POI-Dateien
    
    Args:
        batch_idx: Index im Batch
        voted_predictions: Liste von (idx, coords) Tuples
        gt_landmarks: Ground truth landmarks [n_class, 3]
        poi_metadata: Dict mit Listen von poi_path, subject, vertebra, offset, target_indices
        save_dir: Basisverzeichnis zum Speichern
    """
    
    # Hole Metadaten für dieses Sample
    poi_path = poi_metadata['poi_path'][batch_idx]
    subject = poi_metadata['subject'][batch_idx]
    vertebra = poi_metadata['vertebra'][batch_idx]
    offset = poi_metadata['offset']#[batch_idx]  # numpy array [3]

    print(f"DEBUG: poi_path {poi_path}, subject {subject}, vertebra {vertebra}")
    
    # ========== FIX: Flatten offset falls nötig ==========
    if isinstance(offset, np.ndarray):
        offset = offset.flatten()  # [[43, 43, 0]] → [43, 43, 0]
    elif isinstance(offset, (list, tuple)):
        offset = np.array(offset).flatten()

    if torch.is_tensor(vertebra):
        vertebra = vertebra.item()  # Tensor → Python int
    else:
        vertebra = int(vertebra)  # Sicherheitshalber

    #target_indices = poi_metadata['target_indices']#[batch_idx]  # numpy array der gültigen POI-Indizes
    target_indices = np.array([
            81, 82, 83, 84, 85, 86, 87, 88, 89,
            101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 112, 113, 114, 115, 116,
            117, 118, 119, 120, 121, 122, 123, 124,
            125, 127
        ])
    
    # Lade Original-POI für Metadaten
    original_poi = POI.load(poi_path)

    print(f"\n{'='*60}")
    print(f"POI origin: {original_poi.origin}")
    print(f"POI zoom: {original_poi.zoom}")
    print(f"POI shape: {original_poi.shape}")
    print(f"Padding offset: {offset}")

    origin = original_poi.origin
    rotation = original_poi.rotation
    shape = original_poi.shape
    zoom = original_poi.zoom
    orientation = original_poi.orientation

    
    # Erstelle Save-Verzeichnisse
    os.makedirs(os.path.join(save_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "ground_truth"), exist_ok=True)
    
    # ==== PREDICTIONS SPEICHERN ====
    pred_coords = np.array([coords for idx, coords in voted_predictions])

    valid_mask = np.mean(gt_landmarks, axis=1) > 0

    pred_coords_valid = pred_coords[valid_mask]
    pred_indices_valid = target_indices[valid_mask]  # Nehme an, voted_predictions hat gleiche Reihenfolge wie target_indices
    gt_landmarks_valid = gt_landmarks[valid_mask]

    restored = np.array(pred_coords_valid, dtype=float)

    crop_begin = np.array(crop_begin, dtype=float).squeeze()  # (1,3) → (3,)
    crop_rate = np.array(crop_rate, dtype=float).squeeze()

    print(f"DEBUG: restored shape: {restored.shape}")
    print(f"DEBUG: crop_begin: {crop_begin}, crop_rate: {crop_rate}")
    
    # Inverse: erst begin addieren (undo crop), dann durch rate teilen (undo zoom)
    restored[:, 0] = (restored[:, 0] + crop_begin[0]) / crop_rate[0]
    restored[:, 1] = (restored[:, 1] + crop_begin[1]) / crop_rate[1]
    restored[:, 2] = (restored[:, 2] + crop_begin[2]) / crop_rate[2]




    if len(pred_coords_valid) > 0:
        pred_poi_valid = np_to_ctd(
            restored,#pred_coords_valid,
            vertebra,
            origin,
            rotation,
            idx_list=pred_indices_valid,
            shape=shape,
            zoom=zoom,
            offset=offset,
            orientation=orientation
        )
    
        pred_save_path = os.path.join(
            save_dir, "predictions", 
            f"{subject}_{vertebra}_pred.json"
        )
        pred_poi_valid.save(pred_save_path, verbose=False)

        # Speichere auch globale Version
        pred_global_path = pred_save_path.replace("_pred.json", "_pred_global.json")
        pred_poi_valid.to_global().save_mrk(pred_global_path)

        
        
    # ==== GROUND TRUTH SPEICHERN ====
    #gt_save_path = os.path.join(
    #        save_dir, "ground_truth", 
    #        f"{subject}_{vertebra}_gt_global.json"
    #    )
    
    #original_poi.to_global().save_mrk(gt_save_path)
    
    
    gt_coords = gt_landmarks[valid_mask]
    gt_indices = target_indices[valid_mask]


    gt_restored = np.array(gt_coords, dtype=float)
    crop_begin = np.array(crop_begin, dtype=float)
    crop_rate = np.array(crop_rate, dtype=float)

    
    # Inverse: erst begin addieren (undo crop), dann durch rate teilen (undo zoom)
    gt_restored[:, 0] = (gt_restored[:, 0] + crop_begin[0]) / crop_rate[0]
    gt_restored[:, 1] = (gt_restored[:, 1] + crop_begin[1]) / crop_rate[1]
    gt_restored[:, 2] = (gt_restored[:, 2] + crop_begin[2]) / crop_rate[2]
    

    print(f"DEBUG: gt_coords: {gt_coords}")
    print(f"DEBUG: gt_restored: {gt_restored}")   

    
    if len(gt_coords) > 0:
        gt_poi = np_to_ctd(
            gt_restored,
            vertebra,
            origin,
            rotation,
            idx_list=gt_indices,
            shape=shape,
            zoom=zoom,
            offset=offset,
            orientation=orientation
        )
        
        gt_save_path = os.path.join(
            save_dir, "ground_truth",
            f"{subject}_{vertebra}_gt.json"
        )
        gt_poi.save(gt_save_path, verbose=False)
        
        # Speichere auch globale Version
        gt_global_path = gt_save_path.replace("_gt.json", "_gt_global.json")
        gt_poi.to_global().save_mrk(gt_global_path)


        ########## DEBUG 

        

        original_poi_coords = [
            (
                np.array((-1, -1, -1))
                if not (vertebra, p_idx) in original_poi.keys()
                else np.array(original_poi.centroids[vertebra, p_idx])
            )
            for p_idx in target_indices
        ]

        gt_poi_coords = [
            (
                np.array((-1, -1, -1))
                if not (vertebra, p_idx) in gt_poi.keys()
                else np.array(gt_poi.centroids[vertebra, p_idx])
            )
            for p_idx in target_indices
        ]

        print(f"DEBUG: gt_poi_coords: {gt_poi_coords}" )
        print(f"\nDEBUG: original_poi_coords: {original_poi_coords}" )

        diff = np.array(gt_poi_coords) - np.array(original_poi_coords)
        print(f"DEBUG: diff: {diff}")


    


    

def np_to_ctd(
    t,
    vertebra,
    origin,
    rotation,
    idx_list=None,
    shape=(128, 128, 96),
    zoom=(1, 1, 1),
    offset=(0, 0, 0),
    orientation=None,  # <- Neu: orientation als Argument
):
    ctd = {}

    for i, coords in enumerate(t):
        coords = np.array(coords).astype(float) - np.array(offset).astype(float)
        coords = (coords[0], coords[1], coords[2])
        if idx_list is None:
            ctd[vertebra, i] = coords
        elif i < len(idx_list):
            ctd[vertebra, idx_list[i]] = coords
    
    ###
    if orientation is None:
        raise ValueError("You must provide the orientation of the input POI.")

    ctd = POI(
        centroids=ctd,
        orientation=orientation,
        zoom=zoom,
        shape=shape,
        origin=origin,
        rotation=rotation,
    )

    # ctd.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_((1, 1, 1), verbose=False)

    return ctd
            