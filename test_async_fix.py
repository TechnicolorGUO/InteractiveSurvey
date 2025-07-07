#!/usr/bin/env python
"""
测试异步修复是否有效的简单脚本
"""
import requests
import time
import json

def test_async_upload():
    """测试异步上传功能"""
    
    base_url = "http://localhost:8000"
    
    print("=== 测试异步文件上传功能 ===")
    print("1. 测试简单异步任务...")
    
    # 首先测试简单的异步任务
    try:
        response = requests.post(f"{base_url}/test_async_simple/")
        if response.status_code == 200:
            result = response.json()
            operation_id = result.get('operation_id')
            print(f"✅ 简单异步任务启动成功，operation_id: {operation_id}")
            
            # 轮询进度
            max_polls = 10
            poll_count = 0
            
            while poll_count < max_polls:
                poll_count += 1
                progress_response = requests.get(f"{base_url}/get_operation_progress/", 
                                               params={'operation_id': operation_id})
                
                if progress_response.status_code == 200:
                    progress_data = progress_response.json()
                    progress = progress_data.get('progress', 0)
                    status = progress_data.get('status', 'running')
                    
                    print(f"[{poll_count:2d}] 进度: {progress:3d}% | 状态: {status}")
                    
                    if status == 'completed':
                        print("✅ 简单异步任务测试成功！")
                        break
                    elif status == 'failed':
                        print("❌ 简单异步任务测试失败！")
                        return False
                    else:
                        time.sleep(1)
                else:
                    print(f"❌ 进度查询失败，状态码: {progress_response.status_code}")
                    return False
            
            if poll_count >= max_polls:
                print("⚠️  简单异步任务测试超时")
                return False
                
        else:
            print(f"❌ 简单异步任务启动失败，状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 简单异步任务测试异常: {e}")
        return False
    
    print("\n2. 测试文件上传异步任务...")
    
    # 测试文件上传（不实际上传文件，只测试接口响应）
    try:
        upload_data = {
            'topic': 'Test Topic',
            'mode': 'test',
            'survey_id': 'new',
            'custom_survey_id': 'test_async_' + str(int(time.time()))
        }
        
        response = requests.post(f"{base_url}/upload_refs", data=upload_data)
        
        if response.status_code == 200:
            result = response.json()
            operation_id = result.get('operation_id')
            print(f"✅ 文件上传异步任务启动成功，operation_id: {operation_id}")
            
            # 检查任务状态
            progress_response = requests.get(f"{base_url}/get_operation_progress/", 
                                           params={'operation_id': operation_id})
            
            if progress_response.status_code == 200:
                progress_data = progress_response.json()
                status = progress_data.get('status', 'unknown')
                print(f"✅ 任务状态检查成功，状态: {status}")
                
                if status in ['running', 'completed', 'failed']:
                    print("✅ 异步任务状态正常")
                    return True
                else:
                    print(f"⚠️  任务状态异常: {status}")
                    return False
            else:
                print(f"❌ 任务状态检查失败，状态码: {progress_response.status_code}")
                return False
                
        else:
            print(f"❌ 文件上传异步任务启动失败，状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 文件上传异步任务测试异常: {e}")
        return False

def main():
    print("开始测试异步修复...")
    print("请确保Django服务器正在运行在 http://localhost:8000")
    print("=" * 50)
    
    try:
        success = test_async_upload()
        
        if success:
            print("\n" + "=" * 50)
            print("✅ 所有异步测试通过！")
            print("异步修复应该已经生效。")
        else:
            print("\n" + "=" * 50)
            print("❌ 异步测试失败！")
            print("请检查服务器日志以获取更多信息。")
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
    
    print("\n测试完成")

if __name__ == "__main__":
    main() 