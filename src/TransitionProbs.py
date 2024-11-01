import torch
from tqdm.autonotebook import tqdm
from get_wav import getWavCoeffs

def getTransProbs(dataset, base_restart, device, dense, window_size, freq_balance, time_travel_scale):
    all_low_coeffs = []
    all_high_coeffs = []

    for graph in tqdm(dataset, desc='Processing snapshots'):
        At = graph.adj(ctx=graph.device)
        if dense:
            At = At.to_dense()
        Ft = graph.ndata['X']

        low_coeff, high_coeff, _ = getWavCoeffs(Ft, At, nScales=6, signal_type='digital')
        all_low_coeffs.append(low_coeff)
        all_high_coeffs.append(high_coeff)

    all_low_coeffs = torch.stack(all_low_coeffs)
    all_high_coeffs = torch.stack(all_high_coeffs)

    num_snapshots, num_nodes = all_low_coeffs.shape

    all_restart_probs = torch.full((num_snapshots, num_nodes), base_restart, device=device)
    all_time_travel_probs = torch.zeros((num_snapshots, num_nodes), device=device)

    for t in range(num_snapshots):
        if t < window_size:
            continue
        
        # Calculate the rate of change of low-frequency coefficients (using average change within a window)
        low_freq_changes = []
        for i in range(1, window_size):
            low_freq_changes.append(torch.abs(all_low_coeffs[t-i+1] - all_low_coeffs[t-i]))
        low_freq_change = torch.stack(low_freq_changes).mean(dim=0)
        
        # Normalize the low-frequency change rates and high-frequency coefficients
        norm_low_freq_change = (low_freq_change - low_freq_change.mean()) / (low_freq_change.std() + 1e-8)
        norm_high_freq = (all_high_coeffs[t] - all_high_coeffs[t].mean()) / (all_high_coeffs[t].std() + 1e-8)
        
        # "Calculate time travel probability (weighted sum)"
        time_travel_adjust = freq_balance * norm_low_freq_change + (1 - freq_balance) * norm_high_freq
        
        # Normalize again
        time_travel_adjust = (time_travel_adjust - time_travel_adjust.mean()) / (time_travel_adjust.std() + 1e-8)
        
        # Map values to (0, 1) range using sigmoid function and apply time_travel_scale
        all_time_travel_probs[t] = torch.sigmoid(time_travel_adjust) * time_travel_scale

        # print info
        print(f"Snapshot {t}:")
        print(f"  Time travel probs range: [{all_time_travel_probs[t].min().item():.4f}, {all_time_travel_probs[t].max().item():.4f}]")
        print(f"  Time travel probs mean: {all_time_travel_probs[t].mean().item():.4f}")
        print(f"  Time travel probs std: {all_time_travel_probs[t].std().item():.4f}")
        print("------")

    # make sure probs legal
    all_time_travel_probs = torch.clamp(all_time_travel_probs, min=0, max=1-base_restart)

    return all_restart_probs, all_time_travel_probs