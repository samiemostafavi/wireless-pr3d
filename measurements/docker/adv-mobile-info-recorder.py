import subprocess, json, time, os, sys, datetime, gzip

# example: 
# python3 adv-mobile-info-recorder.py 10s 400ms ./network http://10.10.5.1:50000 device=adv01

def get_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    filename = "mni_" + timestamp + ".json.gz"
    return filename

def save_json_gz(data,outputdir_str):
    filename = get_filename()
    fullpath = os.path.join(outputdir_str, filename)
    with gzip.open(fullpath, 'wt', encoding='utf-8') as gzipped_file:
        json.dump(data, gzipped_file, ensure_ascii=False)

def parse_arguments_to_dict(args):
    argument_dict = {}
    for arg in args:
        # Split the argument by '=' to separate key and value
        key_value = arg.split('=')
        if len(key_value) == 2:
            key, value = key_value
            argument_dict[key] = value
        else:
            print(f"Ignoring invalid argument: {arg}")
    return argument_dict

def main():
    args_num = len(sys.argv)
    if args_num < 5:
        print("no duration, sleep_time, output directory, or server address are provided")
        return

    duration_str = sys.argv[1]
    print(f"duration: {duration_str}")
    
    sleep_time_str = sys.argv[2]
    print(f"sleep time: {sleep_time_str}")
    
    outputdir_str = sys.argv[3]
    os.makedirs(outputdir_str, exist_ok=True)
    print(f"output dir: {outputdir_str}")

    server_addr = sys.argv[4]
    print(f"server address: {server_addr}")

    if args_num > 5:
        # add arbitrary key=values to the results table
        command_line_args = sys.argv[5:]
        arguments_dict = parse_arguments_to_dict(command_line_args)
        print(f"arguments_dict: {arguments_dict}")
    else:
        print(f"no key=value arguments")
        arguments_dict = {}


    # example
    # duration_str = 100s
    # sleep_time_str = 400ms
    # should result:
    # duration = 100 # seconds
    # sleep_time = 0.4 # seconds

    seconds_per_unit = {"ms":0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    def convert_to_seconds(s):
        if s[-2:] == "ms":
            return float(s[:-2]) * seconds_per_unit[s[-2:]]
        else:
            return float(s[:-1]) * seconds_per_unit[s[-1]]

    duration = convert_to_seconds(duration_str)
    sleep_time = convert_to_seconds(sleep_time_str)

    meas_duration = 0
    meas_list = []
    t0 = time.time()
    while meas_duration < duration:
        t = time.time_ns()
        res = subprocess.check_output(["curl","-s", server_addr])
        res = res.decode('utf-8').strip()
        #print(res)
        res = json.loads(res)
        res['timestamp'] = t
        
        # Append arbitrary key=value
        if arguments_dict:
            for key in arguments_dict:
                res[key] = arguments_dict[key]

        meas_list.append(res)
        time.sleep(sleep_time)
        meas_duration = time.time()-t0

    print(f"Duration: {meas_duration}")

    for elem in meas_list:
        new_RSRP = float(elem['RSRP'].split()[0])
        elem['RSRP'] = new_RSRP
        new_RSRQ = float(elem['RSRQ'].split()[0])
        elem['RSRQ'] = new_RSRQ

    save_json_gz(meas_list,outputdir_str)

if __name__ == "__main__":
    main()
