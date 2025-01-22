# Import dependencies
import sys
from A.mainA import taskA
from B.mainB import taskB

def main():
    """
    Main function to execute tasks based on command line arguments.
    This function parses command line arguments to determine which tasks to run
    and in which mode (either 'test' or 'train'). If no arguments are provided,
    both tasks are run in 'test' mode by default.
    Command line arguments:
        a=<mode>: Specifies the mode for task A ('test' or 'train').
        b=<mode>: Specifies the mode for task B ('test' or 'train').
    If an invalid mode is provided, it defaults to 'test'.
    Example usage:
        python main.py a=train b=test
    Tasks:
        taskA: Executes task A in the specified mode.
        taskB: Executes task B in the specified mode.
    """
    # Get command line arguments
    args = sys.argv[1:]

    # Set default values
    run_taskA = False
    run_taskB = False
    taskA_mode = "test"
    taskB_mode = "test"

    if args:
        for arg in args:
            if arg.lower().startswith("a="):
                taskA_mode = arg.split("=")[1].strip().lower()
                # Check if mode is valid
                if taskA_mode != "test" and taskA_mode != "train":
                    taskA_mode = "test"
                run_taskA = True
            elif arg.lower().startswith("b="):
                taskB_mode = arg.split("=")[1].strip().lower()
                # Check if mode is valid
                if taskB_mode != "test" and taskB_mode != "train":
                    taskB_mode = "test"
                run_taskB = True
            else: # If tasks aren't valid, run both tasks
                run_taskA = True
                run_taskB = True
    else:
        # If no arguments are provided, run both tasks
        run_taskA = True
        run_taskB = True

    # Execute specific tasks
    if run_taskA:
        print(f"RUNNING TASK A IN {taskA_mode.upper()}\n")
        taskA(mode=taskA_mode)
    if run_taskB:
        print(f"RUNNING TASK B IN {taskA_mode.upper()}\n")
        taskB(mode=taskB_mode)

if __name__ == "__main__":
    main()
