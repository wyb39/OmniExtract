import json
import os
from datetime import datetime
from typing import Callable, Dict, Any
import traceback
from process_manager import process_manager


def run_task_with_logging(
    callable_obj: Callable[[Dict[str, Any]], Any],
    data: Dict[str, Any],
    name: str,
    log_path: str
) -> Dict[str, Any]:
    """
    Run a task and log to a JSON file
    
    Args:
        callable_obj: The callable object to run
        data: Dictionary parameters to pass to callable_obj
        name: Task name
        log_path: Log file path
    
    Returns:
        Dictionary containing task results and status
    """
    
    # Create initial log record
    log_data = {
        "name": name,
        "data": data,
        "status": "running",
        "created_time": datetime.now().isoformat(),
        "modified_time": datetime.now().isoformat(),
        "error": None,
        "result": None
    }
    
    # Write initial log
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    try:
        # Run the callable object
        result = callable_obj(data)
        
        # Update success status
        log_data.update({
            "status": "succeed",
            "modified_time": datetime.now().isoformat(),
            "result": result,
            "error": None
        })
        
        # Write updated log
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "succeed",
            "result": result,
            "log_path": log_path
        }
        
    except Exception as e:
        # Update failure status
        log_data.update({
            "status": "failed",
            "modified_time": datetime.now().isoformat(),
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        })
        
        # Write updated log
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "failed",
            "error": str(e),
            "log_path": log_path
        }


def cancel_all_running_tasks(runs_dir: str = "gui/runs") -> int:
    """
    Cancel all running tasks in the runs directory by setting their status to 'cancelled'.
    
    Args:
        runs_dir: Path to the runs directory (relative to project root)
    
    Returns:
        Number of tasks cancelled
    """
    # Convert to absolute path if needed
    if not os.path.isabs(runs_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        runs_dir = os.path.join(project_root, runs_dir)
    
    cancelled_count = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(runs_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                try:
                    # Read the JSON file
                    with open(json_path, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    
                    # Check if status is running
                    if log_data.get("status") == "running":
                        # Update to cancelled status with error info
                        log_data.update({
                            "status": "cancelled",
                            "modified_time": datetime.now().isoformat(),
                            "error": {
                                "type": "cancelled",
                                "message": "this task was cancelled because the service was stopped",
                                "traceback": ""
                            }
                        })
                        
                        # Write back the updated data
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(log_data, f, ensure_ascii=False, indent=2)
                        
                        cancelled_count += 1
                        
                except (FileNotFoundError, json.JSONDecodeError, IOError):
                    # Skip files that can't be read or parsed
                    continue
    
    return cancelled_count


def cancel_task(log_path: str) -> bool:
    """
    Cancel a task and update status to cancelled
    
    Args:
        log_path: Log file path
    
    Returns:
        Whether the task was successfully cancelled
    """
    try:
        # Read existing log
        with open(log_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        print(log_data)
        print(log_path)
        if log_data.get("status") == "running":
            process_manager.terminate(log_path)
            log_data.update({
                "status": "cancelled",
                "modified_time": datetime.now().isoformat()
            })
            
            # Write updated log
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            return True
        
        return False
        
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def create_initial_log(name: str, data: Dict[str, Any], log_path: str) -> Dict[str, Any]:
    """
    Create initial log record and save to file
    
    Args:
        name: Task name
        data: Dictionary parameters for the task
        log_path: Log file path
    
    Returns:
        Initial log data dictionary
    """
    log_data = {
        "name": name,
        "data": data,
        "status": "running",
        "created_time": datetime.now().isoformat(),
        "modified_time": datetime.now().isoformat(),
        "error": None,
        "result": None
    }
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    return log_data


def run_task_and_update_log(callable_obj: Callable[[Dict[str, Any]], Any], data: Dict[str, Any], log_path: str) -> Dict[str, Any]:
    """
    Run a task and update the log with results or errors
    
    Args:
        callable_obj: The callable object to run
        data: Dictionary parameters to pass to callable_obj
        log_path: Log file path
    
    Returns:
        Dictionary containing task results and status
    """
    # Read existing log
    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    try:
        # Run the callable object
        result = callable_obj(data)
        
        # Update success status
        log_data.update({
            "status": "succeed",
            "modified_time": datetime.now().isoformat(),
            "result": result,
            "error": None
        })
        
        # Write updated log
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "succeed",
            "result": result,
            "log_path": log_path
        }
        
    except Exception as e:
        # Update failure status
        log_data.update({
            "status": "failed",
            "modified_time": datetime.now().isoformat(),
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        })
        
        # Write updated log
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "failed",
            "error": str(e),
            "log_path": log_path
        }

