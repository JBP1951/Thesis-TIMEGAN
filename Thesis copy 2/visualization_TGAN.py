"""
visualization_TGAN.py
Adaptado mínimamente del TimeGAN original (Yoon et al., NeurIPS 2019)
para funcionar con listas numpy de distinta longitud (caso Dario).

Referencia original:
https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualization(ori_data, generated_data, analysis):
    """Using PCA or tSNE for generated and original data visualization.
    
    Args:
      - ori_data: original data (list or array)
      - generated_data: generated synthetic data (list or array)
      - analysis: 'tsne' or 'pca'
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data), len(generated_data)])
    idx = np.random.permutation(anal_sample_no)
      
    # Convertir a numpy arrays si son listas
    ori_data = np.asarray(ori_data, dtype=object)
    generated_data = np.asarray(generated_data, dtype=object)

    # Seleccionar subconjunto aleatorio
    ori_data = [ori_data[i] for i in idx]
    generated_data = [generated_data[i] for i in idx]

    # Asegurar misma longitud (recortar al mínimo común)
    min_len = min(min(seq.shape[0] for seq in ori_data),
                  min(seq.shape[0] for seq in generated_data))
    dim = ori_data[0].shape[1]

    ori_data = np.array([seq[:min_len, :] for seq in ori_data])
    generated_data = np.array([seq[:min_len, :] for seq in generated_data])

    no, seq_len, dim = ori_data.shape  

    # Promediar en la dimensión de características
    prep_data = np.array([np.mean(seq, axis=1) for seq in ori_data])
    prep_data_hat = np.array([np.mean(seq, axis=1) for seq in generated_data])
      
    # Visualization parameter        
    colors = ["red" for _ in range(anal_sample_no)] + ["blue" for _ in range(anal_sample_no)]    
    
    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)
        
        # Plotting
        f, ax = plt.subplots(1, figsize=(6,5))    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c="red", alpha=0.3, label="Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c="blue", alpha=0.3, label="Sintético")
        ax.legend()  
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y-pca')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    elif analysis == 'tsne':
        # t-SNE Analysis
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        print("🌀 Ejecutando t-SNE (puede tardar unos segundos)...")
        
        n_samples = prep_data_final.shape[0]
        perp = min(40, max(2, n_samples // 3))
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = perp, n_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)
        
        f, ax = plt.subplots(1, figsize=(6,5))
        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                    c="red", alpha=0.3, label="Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                    c="blue", alpha=0.3, label="Sintético")
        ax.legend()
        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y-tsne')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
