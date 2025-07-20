"""
System limits management and monitoring utilities
"""
import os
import resource
import psutil
import subprocess
import platform

def get_current_file_descriptors():
    """Get current number of open file descriptors for this process"""
    try:
        process = psutil.Process()
        return process.num_fds() if hasattr(process, 'num_fds') else len(process.open_files())
    except:
        return -1

def get_file_descriptor_limits():
    """Get soft and hard limits for file descriptors"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        return soft, hard
    except:
        return -1, -1

def set_file_descriptor_limit(limit=4096):
    """
    Attempt to increase the file descriptor limit
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Current limits: soft={soft}, hard={hard}")
        
        # Set soft limit to the requested value or hard limit, whichever is lower
        new_soft = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        
        # Verify the change
        new_soft_verify, new_hard_verify = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"New limits: soft={new_soft_verify}, hard={new_hard_verify}")
        return True
        
    except Exception as e:
        print(f"Failed to set file descriptor limit: {e}")
        return False

def monitor_file_descriptors():
    """Monitor and log current file descriptor usage"""
    current_fds = get_current_file_descriptors()
    soft_limit, hard_limit = get_file_descriptor_limits()
    
    print(f"File Descriptor Status:")
    print(f"  Current FDs in use: {current_fds}")
    print(f"  Soft limit: {soft_limit}")
    print(f"  Hard limit: {hard_limit}")
    print(f"  Usage percentage: {(current_fds/soft_limit)*100:.1f}%" if current_fds > 0 and soft_limit > 0 else "  Usage percentage: unknown")
    
    # Warning if usage is high
    if current_fds > 0 and soft_limit > 0:
        usage_percent = (current_fds / soft_limit) * 100
        if usage_percent > 80:
            print(f"  ⚠️  WARNING: High file descriptor usage ({usage_percent:.1f}%)")
        elif usage_percent > 60:
            print(f"  ⚠️  CAUTION: Moderate file descriptor usage ({usage_percent:.1f}%)")

def check_system_requirements():
    """Check system configuration and suggest improvements"""
    print("System Requirements Check:")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    # Check file descriptor limits
    soft, hard = get_file_descriptor_limits()
    print(f"File descriptor limits: soft={soft}, hard={hard}")
    
    if soft < 1024:
        print("❌ File descriptor soft limit is too low (recommended: ≥1024)")
        print("   Try running: ulimit -n 4096")
    elif soft < 4096:
        print("⚠️  File descriptor soft limit could be higher (recommended: ≥4096)")
    else:
        print("✅ File descriptor limits look good")
    
    # Check available memory
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / (1024**3):.2f} GB / {memory.total / (1024**3):.2f} GB")
    
    if memory.percent > 90:
        print("❌ Very high memory usage")
    elif memory.percent > 75:
        print("⚠️  High memory usage")
    else:
        print("✅ Memory usage looks good")

def suggest_fixes():
    """Provide system-level fix suggestions"""
    print("\n" + "="*50)
    print("SUGGESTED FIXES FOR 'Too many open files' ERROR")
    print("="*50)
    
    print("\n1. Increase file descriptor limits:")
    print("   Temporary (current session):")
    print("   ulimit -n 4096")
    
    print("\n   Permanent (add to ~/.bashrc or ~/.profile):")
    print("   echo 'ulimit -n 4096' >> ~/.bashrc")
    
    print("\n2. System-wide limits (requires root):")
    print("   Edit /etc/security/limits.conf and add:")
    print("   * soft nofile 4096")
    print("   * hard nofile 8192")
    
    print("\n3. For systemd services:")
    print("   Add to service file:")
    print("   [Service]")
    print("   LimitNOFILE=4096")
    
    print("\n4. For Docker containers:")
    print("   docker run --ulimit nofile=4096:8192 ...")
    
    print("\n5. Check for resource leaks in the application:")
    print("   - Ensure OpenAI clients are reused")
    print("   - Ensure database connections are properly closed")
    print("   - Monitor file descriptor usage")
    
    print("\n6. Restart the Django development server after making changes")

def initialize_limits():
    """Initialize system limits when the module is imported"""
    try:
        # Try to set a reasonable limit
        set_file_descriptor_limit(4096)
        print("✅ File descriptor limits increased")
    except Exception as e:
        print(f"⚠️  Could not increase file descriptor limits: {e}")
        print("   You may need to set limits manually")

# Automatically run when module is imported
if __name__ == "__main__":
    check_system_requirements()
    suggest_fixes()
else:
    # Only initialize limits when imported, not when run directly
    initialize_limits() 