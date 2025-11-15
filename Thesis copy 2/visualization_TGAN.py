import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualization(ori_data, generated_data, analysis):
    """Using PCA or tSNE for generated and original data visualization."""

    # Sample size
    anal_sample_no = min([1000, len(ori_data), len(generated_data)])
    idx = np.random.permutation(anal_sample_no)
      
    # Convertir a numpy arrays si son listas
    ori_data = np.asarray(ori_data, dtype=object)
    generated_data = np.asarray(generated_data, dtype=object)

    # Subconjunto aleatorio
    ori_data = [ori_data[i] for i in idx]
    generated_data = [generated_data[i] for i in idx]

    # Asegurar misma longitud (recortar)
    min_len = min(min(seq.shape[0] for seq in ori_data),
                  min(seq.shape[0] for seq in generated_data))

    ori_data = np.array([seq[:min_len, :] for seq in ori_data])
    generated_data = np.array([seq[:min_len, :] for seq in generated_data])

    no, seq_len, dim = ori_data.shape

    # ======================================================
    # CORRECCIÓN CRÍTICA → USAR TODAS LAS VARIABLES (FLATTEN)
    # ======================================================
    prep_data = ori_data.reshape(no, -1)
    prep_data_hat = generated_data.reshape(no, -1)
    # ======================================================

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)

        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)
        
        # Plotting
        plt.figure(figsize=(6,5))
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c="red", alpha=0.3, label="Real")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1],
                    c="blue", alpha=0.3, label="Sintético")
        plt.legend()
        plt.title('PCA: Real vs Sintético')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    elif analysis == 'tsne':
        # t-SNE Analysis
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        print("🌀 Ejecutando t-SNE (puede tardar unos segundos)...")
        
        n_samples = prep_data_final.shape[0]
        perp = min(40, max(5, n_samples // 3))

        tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=500)
        tsne_results = tsne.fit_transform(prep_data_final)
        
        plt.figure(figsize=(6,5))
        plt.scatter(tsne_results[:no,0],  tsne_results[:no,1], 
                    c="red", alpha=0.3, label="Real")
        plt.scatter(tsne_results[no:,0], tsne_results[no:,1], 
                    c="blue", alpha=0.3, label="Sintético")
        plt.legend()
        plt.title('t-SNE: Real vs Sintético')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
