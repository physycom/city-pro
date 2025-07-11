import subprocess
from multiprocessing import Pool
import argparse
# List of commands to be executed in parallel
#commands = [
#    "./build_release/city-pro ./bologna_mdt_detailed/2022-05-12/config_bologna.json",
#    "./build_release/city-pro ./bologna_mdt_detailed/2022-07-01/config_bologna.json",
#    "./build_release/city-pro ./bologna_mdt_detailed/2022-08-05/config_bologna.json",
#    "./build_release/city-pro ./bologna_mdt_detailed/2022-11-11/config_bologna.json",
#    "./build_release/city-pro ./bologna_mdt_detailed/2022-12-30/config_bologna.json",
#    "./build_release/city-pro ./bologna_mdt_detailed/2022-12-31/config_bologna.json",
#    "./build_release/city-pro ./bologna_mdt_detailed/2022-01-31/config_bologna.json",
#    "./build_release/city-pro ./bologna_mdt_detailed/2023-01-01/config_bologna.json",
#    "./build_release/city-pro ./bologna_mdt_detailed/2023-03-18/config_bologna.json"
#]

build_command = "${WORKSPACE}/city-pro/ccm/build.ps1 -UseVCPKG -DisableInteractive -DoNotUpdateTOOL -DoNotDeleteBuildFolder"

commands = [
#    "./build_release/city-pro ./bologna_mdt_center/2022-05-12/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt_center/2022-07-01/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt_center/2022-08-05/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt_center/2022-11-11/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt_center/2022-12-30/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt_center/2022-12-31/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt_center/2022-01-31/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt_center/2023-01-01/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt_center/2023-03-18/config_bologna.json"
]

def build(cmd):
    """Function to execute a command in the shell."""
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}\n{e}")

def run_command(cmd):
    """Function to execute a command in the shell."""
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}\n{e}")
def run_adjust_configuration_file(cmd):
    """Function to execute a command in the shell."""
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}\n{e}")

def run_python_analysis(cmd):
    """Function to execute a command in the shell."""
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}\n{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="configurations bounded to bbox and analysis")
    parser.add_argument("--config_set","-cs",type=str,default="./vars/config/config_days_bbox.json", help="Path to the configuration project with days and bounding box")
    parser.add_argument("--config_analysis","-ca",type=str,default="./vars/config",help="Path to the configuration project with days and bounding box")
    args = parser.parse_args()
    file_config_project = args.config_set
    file_config_analysis = args.config_analysis
    # Create a pool of workers equal to the number of commands
    command_set_right_values_config = f"python3 ./python/work_mdt/SetRightDirectoriesConfiguration.py -c {file_config_project}"
    run_adjust_configuration_file(command_set_right_values_config)
#    build(build_command)
    with Pool(len(commands)) as pool:
        pool.map(run_command, commands)

    cmd_python_analysis = f"python3 ./python/work_mdt/AnalysisPaper.py -c {file_config_analysis}"
    run_python_analysis(cmd_python_analysis)