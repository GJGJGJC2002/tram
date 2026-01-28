#!/usr/bin/env python3
"""
临时修复 fvcore checkpoint.py 中的 PyTorch 2.4.0 兼容性问题
"""

import os
import sys

def patch_fvcore():
    """修补 fvcore 的 checkpoint.py 文件"""
    
    # 找到 fvcore 的安装路径
    try:
        import fvcore
        fvcore_path = os.path.dirname(fvcore.__file__)
        checkpoint_file = os.path.join(fvcore_path, 'common', 'checkpoint.py')
        
        if not os.path.exists(checkpoint_file):
            print(f"错误: 找不到文件 {checkpoint_file}")
            return False
        
        print(f"找到 fvcore checkpoint.py: {checkpoint_file}")
        
        # 备份原文件
        backup_file = checkpoint_file + '.backup'
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy2(checkpoint_file, backup_file)
            print(f"已备份原文件到: {backup_file}")
        
        # 读取文件
        with open(checkpoint_file, 'r') as f:
            content = f.read()
        
        # 检查是否已经打过补丁
        if 'PATCHED_FOR_PYTORCH_2_4' in content:
            print("文件已经打过补丁，无需重复操作")
            return True
        
        # 查找需要修改的代码段
        old_code = """            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)"""
        
        new_code = """            if not isinstance(v, torch.Tensor):
                # PATCHED_FOR_PYTORCH_2_4: Fix compatibility with PyTorch 2.4.0
                import numpy as np
                if isinstance(v, np.ndarray):
                    # Use torch.tensor() instead of torch.from_numpy() for PyTorch 2.4.0 compatibility
                    state_dict[k] = torch.tensor(v)
                else:
                    state_dict[k] = torch.from_numpy(v)"""
        
        if old_code in content:
            content = content.replace(old_code, new_code)
            
            # 写回文件
            with open(checkpoint_file, 'w') as f:
                f.write(content)
            
            print("✅ 补丁应用成功！")
            print("\n修改内容:")
            print("  将 torch.from_numpy(v) 替换为 torch.as_tensor(v)")
            print("\n现在可以重新运行脚本了。")
            return True
        else:
            print("警告: 未找到需要修改的代码段，文件可能已经被修改")
            return False
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def restore_fvcore():
    """恢复 fvcore 的原始文件"""
    try:
        import fvcore
        fvcore_path = os.path.dirname(fvcore.__file__)
        checkpoint_file = os.path.join(fvcore_path, 'common', 'checkpoint.py')
        backup_file = checkpoint_file + '.backup'
        
        if os.path.exists(backup_file):
            import shutil
            shutil.copy2(backup_file, checkpoint_file)
            print(f"✅ 已恢复原始文件")
            return True
        else:
            print("警告: 找不到备份文件")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Patch fvcore for PyTorch 2.4.0 compatibility')
    parser.add_argument('--restore', action='store_true', help='Restore original file from backup')
    args = parser.parse_args()
    
    if args.restore:
        print("恢复 fvcore 原始文件...")
        restore_fvcore()
    else:
        print("为 PyTorch 2.4.0 修补 fvcore...")
        patch_fvcore()

