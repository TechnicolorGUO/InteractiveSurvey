"""
异步任务处理中间件
用于解决Cloudflare 524超时错误
"""
import threading
import time
import traceback
from django.http import JsonResponse
from functools import wraps


class AsyncTaskManager:
    """异步任务管理器"""
    
    def __init__(self):
        self.running_tasks = {}
    
    def start_task(self, task_id, target_func, *args, **kwargs):
        """启动异步任务"""
        if task_id in self.running_tasks:
            return False  # 任务已在运行
        
        def task_wrapper():
            try:
                print(f"[DEBUG] Task {task_id} started in background thread")
                self.running_tasks[task_id] = {'status': 'running', 'start_time': time.time()}
                result = target_func(*args, **kwargs)
                self.running_tasks[task_id]['status'] = 'completed'
                self.running_tasks[task_id]['result'] = result
                print(f"[DEBUG] Task {task_id} completed successfully")
            except Exception as e:
                self.running_tasks[task_id]['status'] = 'failed'
                self.running_tasks[task_id]['error'] = str(e)
                self.running_tasks[task_id]['traceback'] = traceback.format_exc()
                print(f"[DEBUG] Async task {task_id} failed: {e}")
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        thread = threading.Thread(target=task_wrapper)
        thread.daemon = True
        thread.start()
        print(f"[DEBUG] Background thread for task {task_id} started")
        return True
    
    def get_task_status(self, task_id):
        """获取任务状态"""
        return self.running_tasks.get(task_id, {'status': 'not_found'})
    
    def cleanup_old_tasks(self, max_age=3600):
        """清理旧任务（默认1小时）"""
        current_time = time.time()
        to_remove = []
        for task_id, task_info in self.running_tasks.items():
            if current_time - task_info.get('start_time', 0) > max_age:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.running_tasks[task_id]


# 全局任务管理器实例
task_manager = AsyncTaskManager()


def async_task(operation_id_key='operation_id'):
    """
    装饰器：将视图函数转换为异步任务
    
    Args:
        operation_id_key: 用于生成operation_id的参数名
    """
    def decorator(func):
        @wraps(func)
        def wrapper(request, *args, **kwargs):
            # 生成操作ID
            import time
            operation_id = f"{func.__name__}_{int(time.time())}"
            
            # 启动异步任务
            success = task_manager.start_task(
                operation_id, 
                func, 
                request, 
                *args, 
                **kwargs
            )
            
            if not success:
                return JsonResponse({'error': 'Task already running'}, status=409)
            
            # 立即返回operation_id
            return JsonResponse({
                'operation_id': operation_id,
                'status': 'started',
                'message': 'Task started successfully'
            })
        
        return wrapper
    return decorator


class AsyncTaskMiddleware:
    """异步任务中间件"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # 定期清理旧任务
        if hasattr(self, '_last_cleanup'):
            if time.time() - self._last_cleanup > 300:  # 每5分钟清理一次
                task_manager.cleanup_old_tasks()
                self._last_cleanup = time.time()
        else:
            self._last_cleanup = time.time()
        
        response = self.get_response(request)
        return response 