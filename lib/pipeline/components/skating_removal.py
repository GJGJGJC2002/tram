"""SkatingRemovalComponent - 滑步消除组件

对 GVHMR 输出的全局坐标系 SMPL 参数进行后处理：
1. pp_static_joint: 基于静止关节检测修正全局平移（消除脚底滑步）
2. process_ik: 反向运动学修正 body_pose，使修正后的关节位置与修正后的轨迹一致

该组件独立于 HPE Component，便于在两者之间插入像素信息推理等额外处理。
"""

import sys
import os
from typing import Dict, Any
import torch

from lib.pipeline.core.component import Component
from lib.pipeline.core.data import PipelineData


class SkatingRemovalComponent(Component):
    """
    滑步消除组件

    从 GVHMR 预处理组件的输出中读取原始全局参数和静止置信度，
    执行滑步修正和 IK 修正。

    Config:
        gvhmr_root: GVHMR 项目根目录（用于导入后处理函数）
        enable_static_joint: 是否执行静止关节修正 (默认 True)
        enable_ik: 是否执行 IK 修正 (默认 True)
        device: 计算设备
    """

    COMPONENT_TYPE = "skating_removal"

    DEFAULT_CONFIG = {
        'gvhmr_root': 'thirdparty/GVHMR',
        'enable_static_joint': True,
        'enable_ik': True,
        'device': 'cuda',
    }

    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)

        self.gvhmr_root = self.config['gvhmr_root']
        self.enable_static_joint = self.config['enable_static_joint']
        self.enable_ik = self.config['enable_ik']

        self.endecoder = None

    def validate_input(self, data: PipelineData) -> bool:
        """验证输入：需要 GVHMR 的原始输出"""
        if data.smpl_params is None:
            self.logger.warning("No smpl_params found")
            return False

        # 检查 GVHMR 特有字段
        sp = data.smpl_params
        if sp.global_orient_w is None or sp.body_pose_aa is None:
            self.logger.warning(
                "smpl_params missing GVHMR fields (global_orient_w, body_pose_aa). "
                "Ensure GVHMR backend was used for HPE."
            )
            return False

        if sp.static_conf_logits is None:
            self.logger.warning("smpl_params missing static_conf_logits")
            return False

        return True

    def setup(self):
        """初始化 EnDecoder（GVHMR 的编解码器，用于 FK 和 IK）"""
        gvhmr_abs = os.path.abspath(self.gvhmr_root)
        if gvhmr_abs not in sys.path:
            sys.path.insert(0, gvhmr_abs)

        # 确保 hmr4d 从 gvhmr_root 加载（skating_removal 依赖 GVHMR 原版的 EnDecoder）
        # 清除可能由 PromptHMR 版本占据的 hmr4d 模块缓存
        hmr4d_modules = [k for k in sys.modules if k.startswith('hmr4d')]
        for mod_name in hmr4d_modules:
            del sys.modules[mod_name]

        # 确保 gvhmr_root 在 sys.path 最前面，优先于 PromptHMR 版本
        if sys.path[0] != gvhmr_abs:
            if gvhmr_abs in sys.path:
                sys.path.remove(gvhmr_abs)
            sys.path.insert(0, gvhmr_abs)

        old_cwd = os.getcwd()
        os.chdir(gvhmr_abs)

        try:
            from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
            # GVHMR 原版的 EnDecoder 不接受 smplx_path 参数，
            # 它通过 PROJ_ROOT 自动定位 body model 文件
            self.endecoder = EnDecoder().to(self.device)
        finally:
            os.chdir(old_cwd)

        self._is_setup = True
        self.logger.info("SkatingRemoval component initialized")

    def cleanup(self):
        """释放资源"""
        self.endecoder = None
        super().cleanup()

    def execute(self, data: PipelineData) -> PipelineData:
        """执行滑步消除"""
        gvhmr_abs = os.path.abspath(self.gvhmr_root)
        if gvhmr_abs not in sys.path:
            sys.path.insert(0, gvhmr_abs)

        from hmr4d.model.gvhmr.utils.postprocess import pp_static_joint, process_ik

        sp = data.smpl_params

        # 重建 GVHMR outputs 格式（pp_static_joint 和 process_ik 需要的格式）
        # 注意：这些函数期望 batch 维度 (B, L, ...)
        # 缓存加载后字段可能是 numpy 数组，需要转为 tensor
        def _to_tensor(x):
            if x is None:
                return None
            return torch.from_numpy(x).float() if isinstance(x, __import__('numpy').ndarray) else x

        global_orient_w = _to_tensor(sp.global_orient_w)  # (F, 3)
        body_pose_aa = _to_tensor(sp.body_pose_aa)  # (F, 63)
        betas = _to_tensor(sp.betas)  # (F, 10)
        transl_w_raw = _to_tensor(sp.transl_w_raw)  # (F, 3) 原始 world transl
        static_conf_logits = _to_tensor(sp.static_conf_logits)  # (F, J)

        # 添加 batch 维度 -> (1, F, ...)
        outputs = {
            "pred_smpl_params_global": {
                "global_orient": global_orient_w.unsqueeze(0).to(self.device),
                "body_pose": body_pose_aa.unsqueeze(0).to(self.device),
                "betas": betas.unsqueeze(0).to(self.device),
                "transl": transl_w_raw.unsqueeze(0).to(self.device),
            },
            "static_conf_logits": static_conf_logits.unsqueeze(0).to(self.device),
        }

        # 1. 滑步修正 (pp_static_joint)
        if self.enable_static_joint:
            self.logger.info("Running pp_static_joint (skating removal)...")
            post_w_transl = pp_static_joint(outputs, self.endecoder)  # (1, F, 3)
            outputs["pred_smpl_params_global"]["transl"] = post_w_transl
            self.logger.info(
                f"Static joint correction applied. "
                f"transl diff: {(post_w_transl - transl_w_raw.unsqueeze(0).to(self.device)).abs().mean():.6f}"
            )
        else:
            self.logger.info("pp_static_joint disabled, skipping")

        # 2. IK 修正 (process_ik)
        if self.enable_ik:
            self.logger.info("Running process_ik (inverse kinematics correction)...")
            corrected_body_pose = process_ik(outputs, self.endecoder)  # (1, F, 63)
            self.logger.info("IK correction applied")
        else:
            corrected_body_pose = body_pose_aa.unsqueeze(0).to(self.device)
            self.logger.info("process_ik disabled, skipping")

        # 更新 smpl_params
        post_transl_w = outputs["pred_smpl_params_global"]["transl"][0].cpu()  # (F, 3)
        post_body_pose = corrected_body_pose[0].cpu()  # (F, 63)

        # 更新世界坐标系平移（修正后）
        data.smpl_params.global_trans = post_transl_w
        # 更新 body_pose (修正后)
        data.smpl_params.body_pose_aa = post_body_pose

        # 记录元数据
        data.metadata['skating_removal'] = {
            'enable_static_joint': self.enable_static_joint,
            'enable_ik': self.enable_ik,
        }

        self.logger.info("Skating removal complete")
        return data
