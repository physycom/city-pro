import subprocess
from multiprocessing import Pool

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
    "./build_release/city-pro ./bologna_mdt/2022-05-12/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt/2022-07-01/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt/2022-08-05/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt/2022-11-11/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt/2022-12-30/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt/2022-12-31/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt/2022-01-31/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt/2023-01-01/config_bologna.json",
    "./build_release/city-pro ./bologna_mdt/2023-03-18/config_bologna.json"
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
    # Create a pool of workers equal to the number of commands
    command_set_right_values_config = "python3 ./python/SetRightDirectoriesConfiguration.py"
    run_adjust_configuration_file(command_set_right_values_config)
#    build(build_command)
    with Pool(len(commands)) as pool:
        pool.map(run_command, commands)

    cmd_python_analysis = "python3 ./python/work_mdt/script/AnalysisMdt/AnalysisPaper.py"
    run_python_analysis(cmd_python_analysis)