from sklearn.decomposition import PCA
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lmodels import *

my_modules = (vgg,ibln_for_vizu,starlike_vizu,starlike,inverted_bottleneck,ib_for_vizu, cnext_bottleneck,cnext_bottleneck_v2,inverted_bottleneck_cln, cnext_bottleneck_full_norm, sandglass_bottleneck,EdgeNeXt_FeedForward,sandglass_bottleneck_without_norm,cnext_bottleneck_opti,sandglass_vizu)

def analyze_multiple_batches(model, device,dataloader):
    activation_means = {}
    grassmanian_means = {}
    batch_handled = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            module_counters = {}
            
            def hook_fn(module, input, output):
                module_type = module.__class__.__name__
                if module_type not in module_counters:
                    module_counters[module_type] = 0
                else:
                    module_counters[module_type] += 1
                
                module_name = f"{module_type}_{module_counters[module_type]}"
                
                if isinstance(output, torch.Tensor):
                    tensor = output
                elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                    tensor = output[0]
                
                batch, channels, height, width = tensor.shape

                #polar.append(tensor.flatten(-2,-1)[0].permute(1,0).detach().cpu())

                x_flat = tensor.flatten(-2,-1).transpose(1,2)
                x_flat = x_flat / torch.norm(x_flat, dim=2, keepdim=True)
                G = torch.bmm(x_flat, x_flat.transpose(1,2))
                # Step 4: Extract the off-diagonal elements
                G[:, torch.arange(height*width), torch.arange(height*width)] = 0

                if module_name not in activation_means:
                    activation_means[module_name]=G.mean().item()
                else: 
                    activation_means[module_name]+=G.mean().item()
                
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (vgg,ibln_for_vizu,starlike_vizu,starlike,inverted_bottleneck,ib_for_vizu, cnext_bottleneck,cnext_bottleneck_v2,inverted_bottleneck_cln, cnext_bottleneck_full_norm, sandglass_bottleneck,EdgeNeXt_FeedForward,sandglass_bottleneck_without_norm,cnext_bottleneck_opti,sandglass_vizu)):
                    handle = module.register_forward_hook(hook_fn)
                    hooks.append(handle)
            

            inputs  = inputs.to(device)
            outputs = model(inputs)

            for hook in hooks: 
                hook.remove()

            batch_handled += 1

            if batch_handled == 4:
                break
    return activation_means, batch_handled


def analyze_weight_eigenvectors(model, device, dataloader):
    eigenvector_analysis = {}

    def is_target_layer(module):
        target_layer_types = [
            nn.Conv2d, nn.Linear,
        ]
        return any(isinstance(module, layer_type) for layer_type in target_layer_types)
    
    # Function to determine if a Conv2d layer is a pointwise convolution
    def is_pointwise_conv(module):
        return isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1)
    
    with torch.no_grad():
        for batch_idx, (inputs,_) in enumerate(dataloader):
            inputs = inputs.to(device)

            # Reset analysis structures for this batch
            batch_eigenvector_stats = {}
            
            def eigenvector_hook(module, input, output):
                # Only process if it's a target layer
                if not is_target_layer(module):
                    return
                
                # Get module type for tracking
                if isinstance(module, nn.Linear) or is_pointwise_conv(module):
                    module_type = "Linear"
                elif isinstance(module, nn.Conv2d):
                    module_type = "Conv2d"
                else:
                    return  # Skip other module types
                
                if module_type not in batch_eigenvector_stats:
                    batch_eigenvector_stats[module_type] = {
                        'direction_changes': [],
                        'magnitude_changes': []
                    }
                
                if isinstance(input, tuple):
                    x = input[0]
                else:
                    x = input
                
                # Compute weight transformation
                if hasattr(module, 'weight'):
                    # For Conv2d or Linear layers
                    if isinstance(module, nn.Conv2d):
                        # Compute transformed output using the module's forward pass
                        transformed_output = output
                        
                        # Flatten spatial dimensions for analysis
                        x_flat = x.flatten(start_dim=1)
                        transformed_flat = transformed_output.flatten(start_dim=1)
                        
                        # Compute direction change using flattened representations
                        x_norm = x_flat / (x_flat.norm(dim=1, keepdim=True) + 1e-8)
                        transformed_norm = transformed_flat / (transformed_flat.norm(dim=1, keepdim=True) + 1e-8)
                        
                        # Compute cosine similarity across batch
                        direction_changes = torch.diagonal(torch.mm(x_norm.t(), transformed_norm)).mean().item()
                        
                        # Compute magnitude changes
                        magnitude_changes = (transformed_flat.norm(dim=1) / (x_flat.norm(dim=1) + 1e-8)).mean().item()
                    
                    elif isinstance(module, nn.Linear):
                        # For linear layers
                        transformed_output = output
                        
                        # Flatten input and output
                        x_flat = x.flatten(start_dim=1)
                        transformed_flat = transformed_output.flatten(start_dim=1)
                        
                        # Compute direction change
                        x_norm = x_flat / (x_flat.norm(dim=1, keepdim=True) + 1e-8)
                        transformed_norm = transformed_flat / (transformed_flat.norm(dim=1, keepdim=True) + 1e-8)
                        
                        # Compute cosine similarity across batch
                        direction_changes = torch.diagonal(torch.mm(x_norm.t(), transformed_norm)).mean().item()
                        
                        # Compute magnitude changes
                        magnitude_changes = (transformed_flat.norm(dim=1) / (x_flat.norm(dim=1) + 1e-8)).mean().item()

                    # Store results
                    batch_eigenvector_stats[module_type]['direction_changes'].append(direction_changes)
                    batch_eigenvector_stats[module_type]['magnitude_changes'].append(magnitude_changes)

            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (vgg,ibln_for_vizu,starlike_vizu,starlike,inverted_bottleneck,ib_for_vizu, cnext_bottleneck,cnext_bottleneck_v2,inverted_bottleneck_cln, cnext_bottleneck_full_norm, sandglass_bottleneck,EdgeNeXt_FeedForward,sandglass_bottleneck_without_norm,cnext_bottleneck_opti,sandglass_vizu)):
                    for module in model.modules():
                        if is_target_layer(module):
                            hook = module.register_forward_hook(eigenvector_hook)
                            hooks.append(hook)
            
            # Forward pass to trigger hooks
            _ = model(inputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Aggregate results
            for layer_type, stats in batch_eigenvector_stats.items():
                if layer_type not in eigenvector_analysis:
                    eigenvector_analysis[layer_type] = {
                        'direction_changes': [],
                        'magnitude_changes': []
                    }
                
                eigenvector_analysis[layer_type]['direction_changes'].extend(stats['direction_changes'])
                eigenvector_analysis[layer_type]['magnitude_changes'].extend(stats['magnitude_changes'])
            
            # Stop after first batch for demonstration
            break

        # Visualization
        plt.figure(figsize=(15, 5))
        
        # Plot direction changes
        plt.subplot(1, 2, 1)
        plt.title('Direction Changes by Layer Type')
        layer_types = list(eigenvector_analysis.keys())
        direction_boxplot = [
            eigenvector_analysis[layer_type]['direction_changes'] 
            for layer_type in layer_types
        ]
        plt.boxplot(direction_boxplot, labels=layer_types, showfliers=False)
        plt.ylabel('Cosine Similarity')
        plt.xticks(rotation=45)
        
        # Plot magnitude changes
        plt.subplot(1, 2, 2)
        plt.title('Magnitude Changes by Layer Type')
        magnitude_boxplot = [
            eigenvector_analysis[layer_type]['magnitude_changes'] 
            for layer_type in layer_types
        ]
        plt.boxplot(magnitude_boxplot, labels=layer_types, showfliers=False)
        plt.ylabel('Magnitude Ratio')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('weight_eigenvector_analysis.png')
        plt.close()
        
        return eigenvector_analysis


def analyze_weight_eigenvectors_old(model, device, dataloader):
    eigenvector_analysis = {}

    def is_target_layer(module):
        target_layer_types = [
            nn.Conv2d, nn.Linear,
        ]
        return any(isinstance(module, layer_type) for layer_type in target_layer_types)
    
    with torch.no_grad():
        for batch_idx, (inputs,_) in enumerate(dataloader):
            inputs = inputs.to(device)

             # Reset analysis structures for this batch
            batch_eigenvector_stats = {}
            
            def eigenvector_hook(module, input, output):
                # Only process if it's a target layer
                if not is_target_layer(module):
                    return
                
                # Get module name for tracking
                module_name = module.__class__.__name__
                if module_name not in batch_eigenvector_stats:
                    batch_eigenvector_stats[module_name] = {
                        'direction_changes': [],
                        'magnitude_changes': []
                    }
                
                if isinstance(input, tuple):
                    x = input[0]
                else:
                    x = input
                
                # Compute weight transformation
                if hasattr(module, 'weight'):
                    # For Conv2d or Linear layers
                    if isinstance(module, nn.Conv2d):
                        
                        # Compute transformed output using the module's forward pass
                        transformed_output = output
                        
                        # Flatten spatial dimensions for analysis
                        x_flat = x.flatten(start_dim=1)
                        transformed_flat = transformed_output.flatten(start_dim=1)
                        
                        # Compute direction change using flattened representations
                        x_norm = x_flat / (x_flat.norm(dim=1, keepdim=True) + 1e-8)
                        transformed_norm = transformed_flat / (transformed_flat.norm(dim=1, keepdim=True) + 1e-8)
                        
                        # Compute cosine similarity across batch
                        direction_changes = torch.diagonal(torch.mm(x_norm.t(), transformed_norm)).mean().item()
                        
                        # Compute magnitude changes
                        magnitude_changes = (transformed_flat.norm(dim=1) / (x_flat.norm(dim=1) + 1e-8)).mean().item()
                    
                    elif isinstance(module, nn.Linear):
                        # For linear layers
                        transformed_output = output
                        
                        # Flatten input and output
                        x_flat = x.flatten(start_dim=1)
                        transformed_flat = transformed_output.flatten(start_dim=1)
                        
                        # Compute direction change
                        x_norm = x_flat / (x_flat.norm(dim=1, keepdim=True) + 1e-8)
                        transformed_norm = transformed_flat / (transformed_flat.norm(dim=1, keepdim=True) + 1e-8)
                        
                        # Compute cosine similarity across batch
                        direction_changes = torch.diagonal(torch.mm(x_norm.t(), transformed_norm)).mean().item()
                        
                        # Compute magnitude changes
                        magnitude_changes = (transformed_flat.norm(dim=1) / (x_flat.norm(dim=1) + 1e-8)).mean().item()

                    # Store results
                    batch_eigenvector_stats[module_name]['direction_changes'].append(direction_changes)
                    batch_eigenvector_stats[module_name]['magnitude_changes'].append(magnitude_changes)

            hooks = []
            for module in model.modules():
                if is_target_layer(module):
                    hook = module.register_forward_hook(eigenvector_hook)
                    hooks.append(hook)
            
            # Forward pass to trigger hooks
            _ = model(inputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Aggregate results
            for layer, stats in batch_eigenvector_stats.items():
                if layer not in eigenvector_analysis:
                    eigenvector_analysis[layer] = {
                        'direction_changes': [],
                        'magnitude_changes': []
                    }
                
                eigenvector_analysis[layer]['direction_changes'].extend(stats['direction_changes'])
                eigenvector_analysis[layer]['magnitude_changes'].extend(stats['magnitude_changes'])
            
            # Stop after first batch for demonstration
            break

        #print(eigenvector_analysis)
        # Visualization
        plt.figure(figsize=(15, 5))
        
        # Plot direction changes
        plt.subplot(1, 2, 1)
        plt.title('Direction Changes by Layer')
        layer_names = list(eigenvector_analysis.keys())
        direction_boxplot = [
            eigenvector_analysis[layer]['direction_changes'] 
            for layer in layer_names
        ]
        plt.boxplot(direction_boxplot, labels=layer_names,showfliers=False)
        plt.ylabel('Cosine Similarity')
        plt.xticks(rotation=45)
        
        # Plot magnitude changes
        plt.subplot(1, 2, 2)
        plt.title('Magnitude Changes by Layer')
        magnitude_boxplot = [
            eigenvector_analysis[layer]['magnitude_changes'] 
            for layer in layer_names
        ]
        plt.boxplot(magnitude_boxplot, labels=layer_names,showfliers=False)
        plt.ylabel('Magnitude Ratio')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('weight_eigenvector_analysis.png')
        plt.close()
        
        return eigenvector_analysis

def analyze_heads(model, device, dataloader):
    qk = {}
    kv = {}
    qv = {}
    attn_rank = {}
    
    batch_handled = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            module_counters = {}
            
            def hook_fn(module, input, output):
                module_type = module.__class__.__name__
                if module_type not in module_counters:
                    module_counters[module_type] = 0
                else:
                    module_counters[module_type] += 1
                
                module_name = f"{module_type}_{module_counters[module_type]}"
                
                if isinstance(output, torch.Tensor):
                    tensor = output
                elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                    tensor,q,k,v = output
                
                b,p,h,n,d = q.shape
                q_flat = q.reshape(-1, d)  # Shape: (batch_size * patches * heads * tokens, dim_head)
                k_flat = k.reshape(-1, d)
                v_flat = v.reshape(-1, d)

                dots = torch.matmul(q, k.transpose(-1,-2)) * (d**-0.5)
                attn = nn.Softmax(dim=-1)(dots)
                attention_rank = torch.linalg.matrix_rank(attn)

                #G = cos_sim_qk.mean().item() + cos_sim_qv.mean().item() + cos_sim_kv.mean().item()
                if module_name not in qk:
                    qk[module_name]= F.cosine_similarity(q_flat, k_flat, dim=-1).mean().item()  # Cosine similarity between Q and K
                    qv[module_name]= F.cosine_similarity(q_flat, v_flat, dim=-1).mean().item()  # Cosine similarity between Q and V
                    kv[module_name]= F.cosine_similarity(k_flat, v_flat, dim=-1).mean().item()  # Cosine similarity between K and V
                    attn_rank[module_name]=attention_rank.float().mean().item()
                else: 
                    qk[module_name]+= F.cosine_similarity(q_flat, k_flat, dim=-1).mean().item()  # Cosine similarity between Q and K
                    qv[module_name]+= F.cosine_similarity(q_flat, v_flat, dim=-1).mean().item()  # Cosine similarity between Q and V
                    kv[module_name]+= F.cosine_similarity(k_flat, v_flat, dim=-1).mean().item()  # Cosine similarity between K and V
                    attn_rank[module_name]+=attention_rank.float().mean().item()
                
                
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (mvit_scale_dot_product)):
                    handle = module.register_forward_hook(hook_fn)
                    hooks.append(handle)
            

            inputs  = inputs.to(device)
            outputs = model(inputs)

            for hook in hooks: 
                hook.remove()

            batch_handled += 1
    return qk,qv,kv, attn_rank, batch_handled


def analyze_layer_weights(model, layer_names=None):
    """
    Analyze the orthogonality of weights in specified layers of a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to analyze. If None, analyze all Conv2d and Linear layers.
    
    Returns:
        Dictionary mapping layer names to their orthogonality matrices
    """
    results = {}
    
    # Get all layers if not specified
    if layer_names is None:
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_names.append(name)

    # Analyze each layer
    for name, module in model.named_modules():
        if name in layer_names:
            # Extract weights
            if isinstance(module, nn.Conv2d):
                # For Conv2d: [out_channels, in_channels, kernel_h, kernel_w]
                weight = module.weight.data
                
                # Reshape to [out_channels, in_channels*kernel_h*kernel_w]
                weight_flat = weight.reshape(weight.size(0), -1)
                
            elif isinstance(module, nn.Linear):
                # For Linear: [out_features, in_features]
                weight_flat = module.weight.data
            else:
                continue
            
            # Normalize weights
            weight_norm = weight_flat / torch.norm(weight_flat, dim=1, keepdim=True)
            
            # Compute similarity matrix (dot products)
            similarity = torch.mm(weight_norm, weight_norm.t())
            
            # Store result
            results[name] = similarity.cpu().numpy()
    
    return results

def visualize_weight_directions(orthogonality_dict, figsize=(85, 85)):
    """
    Visualize the directional relationships between weight vectors.
    
    Args:
        orthogonality_dict: Dictionary mapping layer names to their orthogonality matrices
    """
    print(orthogonality_dict)
    n_layers = len(orthogonality_dict)
    n_cols = min(3, n_layers)
    print(n_layers, n_cols)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Custom diverging colormap (blue for negative, white for zero, red for positive)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    for i, (name, matrix) in enumerate(orthogonality_dict.items()):
        if i < len(axes):
            # Remove diagonal (self-similarity)
            np.fill_diagonal(matrix, 0)
            
            # Plot heatmap
            im = axes[i].imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
            axes[i].set_title(f"Layer: {name}")
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(orthogonality_dict), len(axes)):
        axes[i].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Direction Similarity', rotation=270, labelpad=20)
    
    plt.suptitle("Weight Vector Directions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig("test.png")
    #plt.show()

# Example usage
def analyze_model_weights(model):
    """
    Analyze and visualize the weight directions in a model.
    
    Args:
        model: PyTorch model to analyze
    """
    # Get orthogonality matrices
    orthogonality_dict = analyze_layer_weights(model)
    
    # Visualize results
    visualize_weight_directions(orthogonality_dict)

def plot_vectors_on_polar(vectors, method="pca", title=None, color_by=None):
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == 'TSNE':
        reducer = TSNE(n_components=2, perplexity=min(20, len(vectors)-1))
    else:
        print("pca or TSNE")

    vectors_2d = reducer.fit_transform(vectors)
    # Convert to polar coordinates
    r = np.sqrt(vectors_2d[:, 0]**2 + vectors_2d[:, 1]**2)
    theta = np.arctan2(vectors_2d[:, 1], vectors_2d[:, 0])
    
    # Convert to degrees
    theta_deg = np.degrees(theta)
    # Ensure all angles are in [0, 360)
    theta_deg = (theta_deg + 360) % 360
    
    return theta_deg, r
    """# Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Plot points
    if color_by is not None:
        scatter = ax.scatter(np.radians(theta_deg), r, c=color_by, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
        cbar.set_label('Value')
    else:
        ax.scatter(np.radians(theta_deg), r, alpha=0.7)
    
    # Set the direction of theta increasing counterclockwise
    ax.set_theta_direction(-1)
    
    # Set the theta zero location to the top
    ax.set_theta_zero_location('N')
    
    # Add grid lines
    ax.grid(True)
    
    # Add title if provided
    if title:
        plt.title(title)
    
    return fig, ax"""

from sklearn.decomposition import PCA
def compute_orthonormal_basis(features, k=None):
    """
    compute orthonormal basis for the subspace spanned by features.

    Args: 
        features: Tensor of shape [C, H*W]
        k: Number of principal components to keep (if None, determined automatically)

    Returns:
        Orthonormal basis of shape [C,k]
    """

    u,s,v = torch.svd(features)
    if k is None:
        #cumulative explained variance
        explained_var = (s**2)/torch.sum(s**2)
        cumulative_var = torch.cumsum(explained_var, dim=0)
        #keep components that explain 95% of variance
        k = torch.sum(cumulative_var < 0.95).item() + 1
        k = min(k, u.shape[1]) #enure k is not larger than the number of components

    #Return the first k columns of U as the orthonomla basis
    return u[:,:k]

def principal_angles(basis_A, basis_B):
    """
    Compute principal angles between two subspaces.

    Args:
        basis_A: Orthonormal basis of shape [n, k_A]
        basis_B: Orthonormal basis of shape [n, k_B]
    Returns:
        Principal angles in radians
    """
    #compute the singular values of the inner porduct matrix
    s = torch.svd(torch.mm(basis_A.t(), basis_B))[1]

    #Clamp values to [-1,1] to avoid numerical issues
    s = torch.clamp(s, -1.0, 1.0)

    #compute angles in radians
    angles = torch.acos(s)
    return angles


def geodesic_distance(basis_A, basis_B):
    """
    Compute the geodesic distance between two subspaces.
    
    Args:
        basis_A: Orthonormal basis of shape [n, k_A]
        basis_B: Orthonormal basis of shape [n, k_B]
        
    Returns:
        Geodesic distance
    """
    angles = principal_angles(basis_A, basis_B)
    return torch.sqrt(torch.sum(angles**2)).item()

def projection_frobenius_distance(basis_A, basis_B):
    """
    Compute the projection Frobenius norm distance between two subspaces.
    
    Args:
        basis_A: Orthonormal basis of shape [n, k_A]
        basis_B: Orthonormal basis of shape [n, k_B]
        
    Returns:
        Projection Frobenius norm distance
    """
    angles = principal_angles(basis_A, basis_B)
    return torch.sqrt(torch.sum(torch.sin(angles)**2)).item() * (1.0 / torch.sqrt(torch.tensor(2.0)))

def chordal_distance(basis_A, basis_B):
    """
    Compute the chordal distance between two subspaces.
    
    Args:
        basis_A: Orthonormal basis of shape [n, k_A]
        basis_B: Orthonormal basis of shape [n, k_B]
        
    Returns:
        Chordal distance
    """
    # Compute projection matrices
    P_A = torch.mm(basis_A, basis_A.t())
    P_B = torch.mm(basis_B, basis_B.t())
    
    # Compute Frobenius norm of the difference
    return torch.norm(P_A - P_B, p='fro').item()


def visualize_principal_angles(angles, name_A, name_B):
    """
    Visualize the distribution of principal angles between subspaces.
    
    Args:
        angles: Tensor of principal angles
        name_A: Name of the first subspace
        name_B: Name of the second subspace
    """
    angles_deg = angles.cpu().numpy() * (180 / np.pi)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(angles_deg)), angles_deg)
    plt.xlabel('Index')
    plt.ylabel('Principal Angle (degrees)')
    plt.title(f'Principal Angles between {name_A} and {name_B}')
    plt.grid(True)
    plt.tight_layout()
    
    # Also plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(angles_deg, bins=20)
    plt.xlabel('Principal Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Principal Angles between {name_A} and {name_B}')
    plt.grid(True)
    plt.tight_layout()


def extract_features_for_grassmannian(features_tensor):
    """
    Prepare features for Grassmannian analysis.
    
    Args:
        features_tensor: Tensor of shape [1, C, H, W]
        
    Returns:
        Reshaped tensor of shape [C, H*W]
    """
    # Remove batch dimension and reshape
    features = features_tensor.squeeze(0)  # [C, H, W]
    C, H, W = features.shape
    features_reshaped = features.reshape(C, H*W)
    
    return features_reshaped








def grassmannian_analysis_for_layer(model, layer_names, input_tensor, k=None):
    """
    Perform Grassmannian analysis for specific layers of a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to analyze
        input_tensor: Input tensor of shape [1, C_in, H_in, W_in]
        k: Number of principal components to keep (if None, determined automatically)
        
    Returns:
        Dictionary with Grassmannian metrics and bases
    """

    results = {}
    features_dict = {}

    hooks = []

    def hook_fn(name):
        def fn(module, input, output):
            features_dict[name] = output.detach()
        return fn
    
    for name in layer_names:
        for n, m in model.named_modules():
            if n == name:
                hooks.append(m.register_forward_hook(hook_fn(name)))
                break

    with torch.no_grad():
        model(input_tensor)
    
    for hook in hooks:
        hook.remove()

    # Process features
    bases = {}
    for name in layer_names:
        features = extract_features_for_grassmannian(features_dict[name])
        basis = compute_orthonormal_basis(features, k)
        bases[name] = basis
        
        # Store some statistics
        results[name] = {
            'feature_shape': features_dict[name].shape,
            'basis_shape': basis.shape,
            'subspace_dim': basis.shape[1]
        }

    distances = {}
    for i, name_A in enumerate(layer_names):
        for j, name_B in enumerate(layer_names):
            if i < j: #Upper traingular matrix
                basis_A = bases[name_A]
                basis_B = bases[name_B]

                #Compute angles
                angles = principal_angles(basis_A, basis_B)

                #Compute distances
                geo_dist = geodesic_distance(basis_A, basis_B)
                proj_dist = projection_frobenius_distance(basis_A, basis_B)
                chord_dist = chordal_distance(basis_A, basis_B)

                distances[(name_A, name_B)] = {
                    'angles': angles,
                    'geodesic': geo_dist,
                    'projection': proj_dist,
                    'chordal': chord_dist
                }
    
    results['distances'] = distances
    results['bases'] = bases

    return results


def compare_models_grassmannian(models_dict, layer_names_dict, input_tensor, file_name="distance_comparison"):
    """
    Compare different models using Grassmannian metrics.
    
    Args:
        models_dict: Dictionary of model name to model
        layer_names_dict: Dictionary of model name to list of layer names
        input_tensor: Input tensor of shape [1, C_in, H_in, W_in]
        
    Returns:
        Dictionary with Grassmannian analysis results
    """
    results = {}

    for model_name, model in models_dict.items():
        print(f"Analyzing {model_name}...")
        layer_names = layer_names_dict[model_name]
        results[model_name] = grassmannian_analysis_for_layer(model, layer_names, input_tensor)

    visualize_models_comparison(results,file_name)
    return results

def visualize_models_comparison(results,file_name):
    """
    Visualize comparison of Grassmannian metrics across models.
    
    Args:
        results: Dictionary with Grassmannian analysis results
    """
    model_names = list(results.keys())
    
    # Prepare data for visualization
    geodesic_distances = {}
    projection_distances = {}
    chordal_distances = {}
    
    for model_name in model_names:
        model_results = results[model_name]
        distances = model_results['distances']
        
        for (layer_A, layer_B), metrics in distances.items():
            key = f"{layer_A} -> {layer_B}"
            
            if key not in geodesic_distances:
                geodesic_distances[key] = {}
                projection_distances[key] = {}
                chordal_distances[key] = {}
            
            geodesic_distances[key][model_name] = metrics['geodesic']
            projection_distances[key][model_name] = metrics['projection']
            chordal_distances[key][model_name] = metrics['chordal']
    
    # Create visualizations
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    
    # Helper function to plot distance comparison
    def plot_distance_comparison(ax, distances_dict, title):
        layers = list(distances_dict.keys())
        x = np.arange(len(layers))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            values = [distances_dict[layer].get(model_name, 0) for layer in layers]
            ax.bar(x + i*width - 0.4 + width/2, values, width, label=model_name)
        
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_ylabel('Distance')
        ax.legend()
        ax.grid(True, axis='y')
    
    plot_distance_comparison(axes[0], geodesic_distances, 'Geodesic Distance Comparison')
    plot_distance_comparison(axes[1], projection_distances, 'Projection Frobenius Distance Comparison')
    plot_distance_comparison(axes[2], chordal_distances, 'Chordal Distance Comparison')
    
    plt.tight_layout()
    plt.savefig(file_name+".png")




def analyze_feature_evolution(model, layer_names, input_tensor, k=None, file_name="feature_evolution"):
    """
    Analyze how feature subspaces evolve through a network.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to analyze (in order)
        input_tensor: Input tensor of shape [1, C_in, H_in, W_in]
        k: Number of principal components to keep (if None, determined automatically)
    """
    results = grassmannian_analysis_for_layer(model, layer_names, input_tensor, k)
    distances = results['distances']
    
    # Create visualization for consecutive layers
    consecutive_distances = {}
    for i in range(len(layer_names)-1):
        layer_A = layer_names[i]
        layer_B = layer_names[i+1]
        key = (layer_A, layer_B)
        
        if key in distances:
            consecutive_distances[f"{layer_A} -> {layer_B}"] = {
                'geodesic': distances[key]['geodesic'],
                'projection': distances[key]['projection'],
                'chordal': distances[key]['chordal']
            }
    
    # Visualize evolution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    keys = list(consecutive_distances.keys())
    x = np.arange(len(keys))
    width = 0.25
    
    ax.bar(x - width, [consecutive_distances[k]['geodesic'] for k in keys], width, label='Geodesic')
    ax.bar(x, [consecutive_distances[k]['projection'] for k in keys], width, label='Projection')
    ax.bar(x + width, [consecutive_distances[k]['chordal'] for k in keys], width, label='Chordal')
    
    ax.set_title('Feature Subspace Evolution Through Network Layers')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_ylabel('Distance')
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(file_name+".png")
    
    return results




def grassmannian_analysis_for_layer_multiple_batches(model, layer_names, input_tensor, k=None):
    """
    Perform Grassmannian analysis for specific layers of a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to analyze
        input_tensor: Input tensor of shape [1, C_in, H_in, W_in]
        k: Number of principal components to keep (if None, determined automatically)
        
    Returns:
        Dictionary with Grassmannian metrics and bases
    """

    results = {}
    features_dict = {}

    hooks = []

    def hook_fn(name):
        def fn(module, input, output):
            features_dict[name] = output.detach()
        return fn
    
    for name in layer_names:
        for n, m in model.named_modules():
            if n == name:
                hooks.append(m.register_forward_hook(hook_fn(name)))
                break

    with torch.no_grad():
        model(input_tensor)
    
    for hook in hooks:
        hook.remove()

    # Process features
    bases = {}
    for name in layer_names:
        B,C,H,W = features_dict[name].shape
        features = features_dict[name].permute(1,0,2,3).reshape(C, B*H*W)
        #features = extract_features_for_grassmannian(features_dict[name])
        basis = compute_orthonormal_basis(features, k)
        bases[name] = basis
        
        # Store some statistics
        results[name] = {
            'feature_shape': features_dict[name].shape,
            'basis_shape': basis.shape,
            'subspace_dim': basis.shape[1]
        }

    distances = {}
    for i, name_A in enumerate(layer_names):
        for j, name_B in enumerate(layer_names):
            if i < j: #Upper traingular matrix
                basis_A = bases[name_A]
                basis_B = bases[name_B]

                #Compute angles
                angles = principal_angles(basis_A, basis_B)

                #Compute distances
                geo_dist = geodesic_distance(basis_A, basis_B)
                proj_dist = projection_frobenius_distance(basis_A, basis_B)
                chord_dist = chordal_distance(basis_A, basis_B)

                distances[(name_A, name_B)] = {
                    'angles': angles,
                    'geodesic': geo_dist,
                    'projection': proj_dist,
                    'chordal': chord_dist
                }
    
    results['distances'] = distances
    results['bases'] = bases

    return results


def analyze_feature_evolution_multiple_batches(model, layer_names, input_tensor, k=None, file_name="feature_evolution"):
    """
    Analyze how feature subspaces evolve through a network.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to analyze (in order)
        input_tensor: Input tensor of shape [B, C_in, H_in, W_in]
        k: Number of principal components to keep (if None, determined automatically)
    """
    results = grassmannian_analysis_for_layer_multiple_batches(model, layer_names, input_tensor, k)
    distances = results['distances']
    
    # Create visualization for consecutive layers
    consecutive_distances = {}
    for i in range(len(layer_names)-1):
        layer_A = layer_names[i]
        layer_B = layer_names[i+1]
        key = (layer_A, layer_B)
        
        if key in distances:
            consecutive_distances[f"{layer_A} -> {layer_B}"] = {
                'geodesic': distances[key]['geodesic'],
                'projection': distances[key]['projection'],
                'chordal': distances[key]['chordal']
            }
    
    # Visualize evolution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    keys = list(consecutive_distances.keys())
    x = np.arange(len(keys))
    width = 0.25
    
    ax.bar(x - width, [consecutive_distances[k]['geodesic'] for k in keys], width, label='Geodesic')
    ax.bar(x, [consecutive_distances[k]['projection'] for k in keys], width, label='Projection')
    ax.bar(x + width, [consecutive_distances[k]['chordal'] for k in keys], width, label='Chordal')
    
    ax.set_title('Feature Subspace Evolution Through Network Layers')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_ylabel('Distance')
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(file_name+"multiple_batches.png")
    
    return results




def analyze_feature_ranks(model, device, dataloader, threshold=1e-5):
    """
    Analyze the rank of feature maps at each layer of the network.
    
    Args:
        model: The neural network model
        device: The device to run the model on
        dataloader: DataLoader providing batches of data
        threshold: Threshold for determining significant singular values
    
    Returns:
        Dictionary mapping layer names to their average feature rank
    """
    rank_means = {}
    batch_handled = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            module_counters = {}
            
            def hook_fn(module, input, output):
                module_type = module.__class__.__name__
                if module_type not in module_counters:
                    module_counters[module_type] = 0
                else:
                    module_counters[module_type] += 1
                
                module_name = f"{module_type}_{module_counters[module_type]}"
                
                if isinstance(output, torch.Tensor):
                    tensor = output
                elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                    tensor = output[0]
                
                batch, channels, height, width = tensor.shape
                
                # Reshape to [batch, height*width, channels]
                x_flat = tensor.flatten(-2,-1).transpose(1,2)
                
                # Compute rank for each example in the batch
                batch_ranks = []
                for i in range(batch):
                    # Get feature map for single example [height*width, channels]
                    feature_map = x_flat[i]
                    
                    # Compute SVD
                    U, S, V = torch.svd(feature_map)
                    
                    # Count singular values above threshold
                    rank = torch.sum(S > threshold).item()
                    batch_ranks.append(rank)
                
                # Compute average rank across the batch
                avg_rank = sum(batch_ranks) / len(batch_ranks)
                
                # Store or update the rank
                if module_name not in rank_means:
                    rank_means[module_name] = avg_rank
                else:
                    rank_means[module_name] += avg_rank
            
            # Register hooks
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (inverted_bottleneck, cnext_bottleneck, cnext_bottleneck_v2, sandglass_bottleneck)):
                    handle = module.register_forward_hook(hook_fn)
                    hooks.append(handle)
            
            # Forward pass
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            batch_handled += 1
    
    # Compute average rank across all batches
    if batch_handled > 0:
        for module_name in rank_means:
            rank_means[module_name] /= batch_handled
    
    return rank_means, batch_handled

def compute_effective_rank(model, device, dataloader, cumulative_energy=0.99):
    """
    Compute the effective rank of feature maps based on singular value energy.
    
    Args:
        model: The neural network model
        device: The device to run the model on
        dataloader: DataLoader providing batches of data
        cumulative_energy: The minimum cumulative energy to preserve (0.99 = 99%)
        
    Returns:
        Dictionary mapping layer names to their average effective rank
    """
    effective_ranks = {}
    max_ranks = {}  # Store maximum possible rank for each layer
    batch_handled = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            module_counters = {}
            
            def hook_fn(module, input, output):
                module_type = module.__class__.__name__
                if module_type not in module_counters:
                    module_counters[module_type] = 0
                else:
                    module_counters[module_type] += 1
                
                module_name = f"{module_type}_{module_counters[module_type]}"
                
                if isinstance(output, torch.Tensor):
                    tensor = output
                elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                    tensor = output[0]
                
                batch, channels, height, width = tensor.shape
                
                # Reshape to [batch, height*width, channels]
                x_flat = tensor.flatten(-2,-1).transpose(1,2)
                
                # Compute maximum possible rank (min of rows and columns)
                max_rank = min(height*width, channels)
                
                # For each example in batch
                batch_effective_ranks = []
                for i in range(batch):
                    # Get feature map for single example [height*width, channels]
                    feature_map = x_flat[i]
                    
                    # Compute SVD
                    U, S, V = torch.svd(feature_map)
                    
                    # Normalize singular values to get energy distribution
                    total_energy = torch.sum(S)
                    if total_energy > 0:  # Avoid division by zero
                        energy_ratio = S / total_energy
                        cumulative_energy_ratio = torch.cumsum(energy_ratio, dim=0)
                        
                        # Find how many singular values needed to reach target energy
                        effective_rank = torch.sum(cumulative_energy_ratio <= cumulative_energy).item() + 1
                        effective_rank = min(effective_rank, max_rank)  # Cap at max possible rank
                    else:
                        effective_rank = 0
                    
                    batch_effective_ranks.append(effective_rank)
                
                # Average effective rank across batch
                avg_effective_rank = sum(batch_effective_ranks) / len(batch_effective_ranks)
                
                # Store or update
                if module_name not in effective_ranks:
                    effective_ranks[module_name] = avg_effective_rank
                    max_ranks[module_name] = max_rank
                else:
                    effective_ranks[module_name] += avg_effective_rank
            
            # Register hooks
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (inverted_bottleneck, cnext_bottleneck, cnext_bottleneck_v2, sandglass_bottleneck)):
                    handle = module.register_forward_hook(hook_fn)
                    hooks.append(handle)
            
            # Forward pass
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            batch_handled += 1
    
    # Compute average effective rank across all batches
    if batch_handled > 0:
        for module_name in effective_ranks:
            effective_ranks[module_name] /= batch_handled
    
    return effective_ranks, max_ranks, batch_handled


def visualize_feature_ranks(ranks_dict, max_ranks_dict=None, effective_ranks_dict=None, figsize=(12, 8)):
    """
    Create a beautiful visualization of feature ranks across different layers.
    
    Args:
        ranks_dict: Dictionary mapping layer names to their rank values
        max_ranks_dict: Optional dictionary of maximum possible ranks for each layer
        effective_ranks_dict: Optional dictionary of effective ranks for each layer
        figsize: Size of the figure (width, height)
    """
    # Set the style
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)
    
    # Process layer names for better display
    layer_names = list(ranks_dict.keys())
    display_names = [name.replace('_', ' ') for name in layer_names]
    
    # Extract layer types and indices for grouping
    layer_types = []
    layer_indices = []
    
    for name in layer_names:
        parts = name.split('_')
        layer_types.append(parts[0])
        layer_indices.append(int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0)
    
    # Get unique layer types for color coding
    unique_types = list(set(layer_types))
    colors = sns.color_palette("viridis", len(unique_types))
    type_color_map = dict(zip(unique_types, colors))
    
    # Create color list for each bar
    bar_colors = [type_color_map[t] for t in layer_types]
    
    # Sort by layer type and index
    sorted_indices = np.lexsort((layer_indices, layer_types))
    sorted_names = [display_names[i] for i in sorted_indices]
    sorted_ranks = [ranks_dict[layer_names[i]] for i in sorted_indices]
    sorted_colors = [bar_colors[i] for i in sorted_indices]
    
    # Create x positions
    x_pos = np.arange(len(sorted_names))
    
    # Create main bar plot
    bars = plt.bar(x_pos, sorted_ranks, color=sorted_colors, alpha=0.7, width=0.6)
    
    # Add max ranks if provided
    if max_ranks_dict:
        sorted_max_ranks = [max_ranks_dict[layer_names[i]] for i in sorted_indices]
        plt.bar(x_pos, sorted_max_ranks, color='none', edgecolor='black', linestyle='--', 
                alpha=0.5, width=0.6, label='Maximum Possible Rank')
    
    # Add effective ranks if provided
    if effective_ranks_dict:
        sorted_effective_ranks = [effective_ranks_dict[layer_names[i]] for i in sorted_indices]
        plt.plot(x_pos, sorted_effective_ranks, 'ro-', label='Effective Rank (99% Energy)', markersize=8)
    
    # Set labels and title
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Rank', fontsize=14)
    plt.title('Feature Map Rank Analysis Across Network Layers', fontsize=16, fontweight='bold')
    
    # Set x-ticks and labels
    plt.xticks(x_pos, sorted_names, rotation=45, ha='right', fontsize=10)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create legend for layer types
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=type_color_map[t], alpha=0.7, label=t) 
                      for t in unique_types]
    
    if max_ranks_dict:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', 
                                           linestyle='--', label='Maximum Possible Rank'))
    
    if effective_ranks_dict:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='red', label='Effective Rank',
                                        markersize=8, linestyle='-'))
    
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    
    # Adjust layout and add padding
    plt.tight_layout()
    
    # Add text annotations for key insights
    if max_ranks_dict:
        utilization = np.mean([r/max_ranks_dict[layer_names[i]] for i, r in enumerate(sorted_ranks)])
        plt.figtext(0.02, 0.02, f"Average Rank Utilization: {utilization:.1%}", 
                   fontsize=10, ha="left")
    
    return plt

# Example usage:
def plot_model_ranks(model, device, dataloader, figname="test.png"):
    """
    Create a comprehensive visualization of feature ranks in a model.
    
    Args:
        model: PyTorch model
        device: Device to run the model on
        dataloader: DataLoader for input data
    """
    # Get ranks
    ranks, _ = analyze_feature_ranks(model, device, dataloader, threshold=1e-5)
    effective_ranks, max_ranks, _ = compute_effective_rank(model, device, dataloader)
    
    # Create visualization
    plt = visualize_feature_ranks(ranks, max_ranks, effective_ranks)
    
    # Add a custom color bar to show rank utilization
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Layer Type', rotation=270, labelpad=20)
    
    plt.savefig(figname)
    
    return ranks, effective_ranks, max_ranks