pca = PCA(n_components=2)
# X_pca = pca.fit_transform(x)

# plt.figure(figsize=(8,6))
# plt.scatter(X_pca[:,0],X_pca[:,1],c = y,cmap="coolwarm",edgecolors='k')
# plt.xlabel("PCA Component 1")      # ป้ายแกนนอน
# plt.ylabel("PCA Component 2")      # ป้ายแกนตั้ง
# plt.title("PCA Visualization")     # หัวข้อกราฟ
# plt.colorbar(label="Class")        # แถบสีสำหรับ class (target)
# plt.show()