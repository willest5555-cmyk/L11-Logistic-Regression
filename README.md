# Iris 鳶尾花分類與 3D PCA 互動式視覺化專案

本專案旨在結合機器學習經典的邏輯斯迴歸（Logistic Regression）分類模型與主成分分析（PCA），對經典的 Iris（鳶尾花）資料集進行分類實作，並透過 **Three.js** 技術建構網頁端 3D 互動式降維視覺化圖表。專案內部的機器學習流程完整依循標準的 **CRISP-DM** 方法論進行設計。

---

## 📁 檔案結構說明

*   [regression_crisp_dm.py](file:///d:/AICourse26/ML/L11%20Logistic%20Regression/regression_crisp_dm.py) - 遵循 CRISP-DM 流程的完整 Iris 分類與評估腳本。
*   [iris_pca_3d_demo.py](file:///d:/AICourse26/ML/L11%20Logistic%20Regression/iris_pca_3d_demo.py) - 執行 PCA 降維（3 主成分）並生成網頁端 3D 繪圖所需之 JSON 資料。
*   [iris_pca_3d_demo.html](file:///d:/AICourse26/ML/L11%20Logistic%20Regression/iris_pca_3d_demo.html) - 基於 Three.js 實作的 3D 互動式散佈圖網頁介面。
*   [package.json](file:///d:/AICourse26/ML/L11%20Logistic%20Regression/package.json) - 定義 Web 端所需之前端依賴（Three.js）。
*   `outputs/` - 專案自動生成的輸出目錄：
    *   `iris_confusion_matrix.png` - 邏輯斯迴歸模型的混淆矩陣圖表。
    *   `iris_pca_points.json` - 由 `iris_pca_3d_demo.py` 生成的 PCA 降維點資料。
    *   `iris_pca_demo_screenshot.png` - 3D 互動式網頁的預覽截圖。

---

## 🛠️ 開發環境與安裝步驟

### 1. Python 環境準備
請確保您的 Python 環境已安裝以下必要的資料科學軟體包：
```bash
pip install pandas scikit-learn matplotlib
```

### 2. Node.js 依賴安裝（用於 3D 網頁）
本專案的 3D 互動式網頁依賴 **Three.js**。請在專案根目錄下執行以下指令以安裝依賴：
```bash
npm install
```

---

## 🚀 使用指南

請依序執行以下步驟以完整體驗本專案：

### 步驟 1：執行 CRISP-DM 機器學習模型
執行以下指令，系統將會載入 Iris 資料集、進行資料處理、訓練邏輯斯迴歸模型、輸出評估指標，並在 `outputs/` 資料夾下存檔混淆矩陣。
```bash
python regression_crisp_dm.py
```

### 步驟 2：生成 PCA 3D 降維點資料
執行以下指令，將 Iris 的四維特徵透過 PCA 降維投影至前三個主成分，並輸出 JSON 點資料至 `outputs/iris_pca_points.json`。
```bash
python iris_pca_3d_demo.py
```

### 步驟 3：啟動本機伺服器並檢視 3D 互動視覺化
因為網頁需要載入本地的 JSON 資料檔，請使用本地伺服器開啟 HTML。您可以透過 Python 輕鬆啟動：
```bash
python -m http.server 8000
```
啟動後，請於瀏覽器輸入以下網址：
👉 [http://127.0.0.1:8000/iris_pca_3d_demo.html](http://127.0.0.1:8000/iris_pca_3d_demo.html)

---

## 📊 專案技術細節

### 一、CRISP-DM 流程實作
在 [regression_crisp_dm.py](file:///d:/AICourse26/ML/L11%20Logistic%20Regression/regression_crisp_dm.py) 中，完整體現了 CRISP-DM 流程：
1.  **Business Understanding（業務理解）**：設定預測目標（區分 Setosa, Versicolor, Virginica 三類鳶尾花）與成功指標（高測試準確度與良好可解釋性）。
2.  **Data Understanding（資料理解）**：分析資料維度、特徵分佈與是否有缺失值（Iris 資料集無缺失值，三類別分佈均勻）。
3.  **Data Preparation（資料準備）**：進行分層隨機切分（Stratified Train/Test Split），以保證類別比例一致。
4.  **Modeling（模型建構）**：建立 `StandardScaler` 與 `LogisticRegression` 的 Pipeline。
5.  **Evaluation（模型評估）**：計算 Test Accuracy（測試準確度），並繪製混淆矩陣儲存為 `iris_confusion_matrix.png`。
6.  **Deployment（部署與測試）**：使用一筆模擬的鳶尾花資料進行即時預測，並輸出各類別預測機率。

### 二、3D PCA 視覺化互動功能
網頁端介面提供以下互動功能：
*   **3D Orbit Controls（軌道旋轉與縮放）**：支援滑鼠左鍵拖曳旋轉、右鍵平移、滾輪縮放。
*   **自動旋轉（Auto rotate）**：可隨時開啟或關閉場景的自動旋轉展示。
*   **點大小調節（Point size）**：動態拉動滑桿調整散佈點的渲染大小。
*   **種類篩選（Legend Toggle）**：透過圖例核取方塊，可單獨顯示或隱藏特定的鳶尾花品種（Setosa/Versicolor/Virginica）。
*   **懸停提示（Tooltip Hover）**：滑鼠懸停於點上方時，會顯示該樣本的編號以及降維後的 PC1、PC2、PC3 精確座標。
