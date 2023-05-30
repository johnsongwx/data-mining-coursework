# 分类、预测与聚类

计算机学院 JohnsonGuo

[toc]

## 作业选择

任务：本作业将通过 K-means 算法对 20 Newsgroups 数据集中的文本数据进行聚类。首先是数据的预处理，提取文本特征；然后将数据转换为特征向量，使用 K-means 算法进行聚类；最后对聚类结果进行评估和可视化分析。

数据集：The 20 Newsgroups data set
数据集介绍：The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. To the best of my knowledge, it was originally collected by Ken Lang, probably for his Newsweeder: Learning to filter netnews paper, though he does not explicitly mention this collection. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.

要求：
1. 数据预处理：去除停用词、数字、符号等，提取文本特征。
2. 特征向量化：将预处理后的文本使用 TF-IDF 方法进行向量化。
3. K-means 模型训练：根据设定的 K 值，构建 K-means 模型，并对向量化的文本数据进行聚类。
4. 聚类结果分析：评估聚类结果，可使用轮廓系数、CH 指数等方法。
5. 可视化：使用降维方法（如 PCA 或 t-SNE）将高维数据降维至2D或3D，然后进行可视化，观察聚类效果。

## 基于K-means的文本聚类

见Jupyter notebook文件。



