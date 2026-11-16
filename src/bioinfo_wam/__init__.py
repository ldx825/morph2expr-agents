"""bioinfo_wam —— 细胞形态 → 转录组 扰动响应预测流水线。

七个智能体模块与任务书一一对应：

    1. agent1_qc          图像/特征质控
    2. agent2_morph       形态特征提取（X_morph）
    3. agent3_expr        转录组预处理（Y_expr）
    4. agent4_pertencode  扰动信息编码
    5. agent5_predict     跨模态预测
    6. agent6_batch       批次效应诊断与校正
    7. agent7_validate    生物学验证
"""

__version__ = "0.1.0"

AGENTS = {
    1: "agent1_qc",
    2: "agent2_morph",
    3: "agent3_expr",
    4: "agent4_pertencode",
    5: "agent5_predict",
    6: "agent6_batch",
    7: "agent7_validate",
}
