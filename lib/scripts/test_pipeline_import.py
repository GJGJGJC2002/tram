#!/usr/bin/env python3
"""测试 Pipeline 模块导入"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

def test_imports():
    """测试所有模块可以正常导入"""
    print("Testing Pipeline module imports...")
    
    # Core
    print("  - Importing core modules...")
    from lib.pipeline.core.data import PipelineData, CameraParams, SMPLParams
    from lib.pipeline.core.component import Component, BackendComponent, Backend
    from lib.pipeline.core.pipeline import Pipeline, IterativePipeline
    print("    ✓ Core modules imported successfully")
    
    # Builder
    print("  - Importing builder...")
    from lib.pipeline.builder import PipelineBuilder
    print("    ✓ Builder imported successfully")
    
    # Hooks
    print("  - Importing hooks...")
    from lib.pipeline import hooks
    print("    ✓ Hooks imported successfully")
    
    # Main package
    print("  - Importing main package...")
    from lib.pipeline import (
        PipelineData,
        Pipeline,
        IterativePipeline,
        PipelineBuilder,
    )
    print("    ✓ Main package imported successfully")
    
    print("\n✓ All imports successful!")
    
    return True


def test_data_structures():
    """测试数据结构"""
    print("\nTesting data structures...")
    
    from lib.pipeline.core.data import PipelineData, CameraParams, SMPLParams
    import numpy as np
    
    # 测试 CameraParams
    cam = CameraParams(
        focal_length=1000.0,
        principal_point=np.array([540, 960])
    )
    print(f"  - CameraParams: focal={cam.focal_length}, center={cam.principal_point}")
    
    # 测试 PipelineData
    data = PipelineData(
        sequence_name="test_seq",
        sequence_path="/path/to/test",
        image_paths=["/path/to/img1.jpg", "/path/to/img2.jpg"],
    )
    print(f"  - PipelineData: {data}")
    print(f"    frames: {data.get_num_frames()}")
    
    # 测试记录阶段
    data.record_stage("detection")
    print(f"    current_stage: {data.current_stage}")
    print(f"    history: {len(data.metadata['history'])} entries")
    
    print("\n✓ Data structures working correctly!")
    return True


def test_pipeline_creation():
    """测试 Pipeline 创建"""
    print("\nTesting pipeline creation...")
    
    from lib.pipeline.core.pipeline import Pipeline
    
    # 创建简单 Pipeline
    pipeline = Pipeline(name="test_pipeline")
    print(f"  - Created pipeline: {pipeline}")
    
    # 测试配置
    pipeline.config['device'] = 'cpu'
    pipeline.config['output_dir'] = 'results/test'
    print(f"  - Config: device={pipeline.config['device']}")
    
    print("\n✓ Pipeline creation working correctly!")
    return True


def test_builder():
    """测试 Builder"""
    print("\nTesting PipelineBuilder...")
    
    from lib.pipeline.builder import PipelineBuilder
    
    # 列出可用组件类型
    types = PipelineBuilder.list_component_types()
    print(f"  - Available component types: {types}")
    
    # 测试从字典创建
    config = {
        'name': 'test_pipeline',
        'device': 'cpu',
        'components': []  # 空组件列表用于测试
    }
    
    pipeline = PipelineBuilder.from_dict(config)
    print(f"  - Created pipeline from dict: {pipeline}")
    
    print("\n✓ Builder working correctly!")
    return True


def main():
    print("=" * 60)
    print("Pipeline Module Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_imports()
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_data_structures()
    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_pipeline_creation()
    except Exception as e:
        print(f"✗ Pipeline creation test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_builder()
    except Exception as e:
        print(f"✗ Builder test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())

