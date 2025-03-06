import os
import csv
from collections import defaultdict
import ipaddress
import heapq
import struct
import numpy as np


# 定义IID类型
IID_TYPES = {
    'IID_MACDERIVED': 0,
    'IID_LOWBYTE': 0,
    'IID_EMBEDDEDIPV4_32': 0,
    'IID_EMBEDDEDIPV4_64': 0,
    'IID_PATTERN_BYTES': 0,
    'IID_RANDOM': 0
}
def load_oui_database(oui_file='oui.csv'):
    """预加载OUI数据库到内存中的字典"""
    oui_db = defaultdict(lambda: ("Not Found", "Not Found"))
    with open(oui_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            oui_key = row['Assignment'].replace('', '').upper()
            oui_db[oui_key] = (row['Organization Name'], row['Organization Address'])
    return oui_db

oui_db = load_oui_database('E:\\科研\\小论文\\ip数据分析\\MAC地址\\oui.csv')

def zero_byte_iid(ipv6_address):
    """计算 IPv6 地址后 64 位中零字节的数量"""
    ip_bytes = ipv6_address.packed[-8:]  # 只取后 64 位
    return ip_bytes.count(b'\x00')

def check_iid_type(ipv6_address):
    """根据IPv6地址部分分类IID类型"""
    # 将IPv6地址转换为字节形式
    ip_bytes = ipv6_address.packed
    s6_addr32 = struct.unpack("!IIII", ip_bytes)

    # 检查每个32位部分
    if (s6_addr32[2] & 0x000000FF) == 0x000000FF and (s6_addr32[3] & 0xFE000000) == 0xFE000000:
        # 如果符合此条件，进一步判断是否为真正的 MAC 衍生
        mac_address = extract_mac_from_eui64(str(ipv6_address))
        if query_oui(mac_address)[0] != 'Not Found':
            return 'IID_Really_MACDERIVED'
        else:
            return 'IID_MACDERIVED'
    elif s6_addr32[2] == 0 and (s6_addr32[3] & 0xFF000000) != 0 and (s6_addr32[3] & 0x0000FFFF) != 0:
        return 'IID_EMBEDDEDIPV4_32'
    elif s6_addr32[2] == 0 and (s6_addr32[3] & 0xFF000000) == 0 and (s6_addr32[3] & 0x0000FFFF) != 0:
        return 'IID_LOWBYTE'
    elif (s6_addr32[2] >> 16) <= 0x255 and (s6_addr32[2] & 0x0000FFFF) <= 0x255 and \
         (s6_addr32[3] >> 16) <= 0x255 and (s6_addr32[3] & 0x0000FFFF) <= 0x255:
        return 'IID_EMBEDDEDIPV4_64'
    elif zero_byte_iid(ipv6_address) > 2:
        return 'IID_PATTERN_BYTES'
    else:
        return 'IID_RANDOM'



def extract_mac_from_eui64(ipv6_str):
    """
    从EUI-64格式字符串转换为MAC地址。
    :param eui64_str: EUI-64格式的字符串，包含'ff:fe'。
    :return: 转换后的MAC地址字符串。
    """
    # 去除EUI-64中的'ff:fe'
    try:
        # 解析IPv6地址
        addr = ipaddress.IPv6Address(ipv6_str)
        # 计算IID的整数表示，直接提取最后64位，但不进行EUI-64特有的位操作
        iid_hex = ''.join(['{:02x}'.format(b) for b in addr.packed[-8:]])
        # 将IID转换为EUI-64格式字符串，以便进行后续处理
        eui64_str = '-'.join(iid_hex[i:i + 2] for i in range(0, len(iid_hex), 2))
        eui64_no_fffe = eui64_str.replace("ff-fe", "", 1)
        eui64_no_fffe = eui64_no_fffe.replace("--", "-", 1)
        # 翻转第七位（从左往右数），在16进制表示中，'e'变为'd'，'d'变为'e'等
        hex_flip_map = {
            '0': '2', '2': '0',
            '1': '3', '3': '1',
            '4': '6', '6': '4',
            '5': '7', '7': '5',
            '8': 'a', 'a': '8',
            '9': 'b', 'b': '9',
            'c': 'e', 'e': 'c',
            'd': 'f', 'f': 'd'
        }
        seventh_bit_flipped = (
            eui64_no_fffe[0] +  # 保留第一个字符不变
            hex_flip_map.get(eui64_no_fffe[1], eui64_no_fffe[1]) +  # 翻转并替换第二个字符
            eui64_no_fffe[2:]  # 保留剩余部分不变
        )
        # 添加冒号分隔符，转换为标准MAC地址格式
        mac_with_colons = "".join(seventh_bit_flipped[i:i + 2] for i in range(0, len(seventh_bit_flipped), 2))
        mac_with_colons_new = mac_with_colons.replace("-", ":")
        return mac_with_colons_new
    except ValueError:
        return None

def query_oui(mac_address):
    # 假设这里有一个预加载的 OUI 数据库
    if mac_address:
        mac_prefix = mac_address.split(':')[0:3]
        mac_prefix_key = ''.join(mac_prefix).upper()
        return oui_db.get(mac_prefix_key, ('Not Found', 'Not Found'))
    return ('Not Found', 'Not Found')


def parse_ipv6_to_48_prefix(ipv6_address):
    """解析IPv6地址并返回/48子网前缀"""
    return str(ipaddress.IPv6Network(ipv6_address + '/48', strict=False))

def calculate_variance(data):
    """计算数据集的方差"""
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / n

def calculate_range(data):
    """计算数据集的最大值和最小值之差"""
    return max(data) - min(data)

def find_second_smallest(data):
    """找到数据集中的第二小值"""
    if len(data) < 2:
        raise ValueError("Data must have at least two elements")
    smallest_two = heapq.nsmallest(2, data)
    return smallest_two[1]

def calculate_percentile(data, percentile):
    """计算数据集的百分位数"""
    return np.percentile(data, percentile)

def calculate_statistics_of_differences(percentiles):
    """计算百分位差的统计特征"""
    differences = [percentiles[i] - percentiles[i-1] for i in range(1, len(percentiles))]
    return {
        'variance_diff_p': calculate_variance(differences),
        'max_diff_p': max(differences),
        'min_diff_p': min(differences),
        '95th_diff_p': calculate_percentile(differences, 95),
        '5th_diff_p': calculate_percentile(differences, 5),
        'mean_diff_p': sum(differences) / len(differences)
    }





def process_csv(input_file, output_file):
    # 存储每个/48子网的信息
    subnet_data = defaultdict(lambda: {
        "rtts": [],
        "count": 0,
        "rtts_type_1_or_3": [],
        "iid_types": defaultdict(int),
        "total_iids": 0
    })

    # 读取CSV文件
    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            subnet = parse_ipv6_to_48_prefix(row['saddr'])
            rtt = float(row['rtt'])
            type_ = row.get('type')  # 获取type字段，如果没有则为None
            if type_ is not None:
                type_ = int(type_)
            subnet_data[subnet]["rtts"].append(rtt)
            subnet_data[subnet]["count"] += 1
            if type_ == 1 or type_ == 3:
                subnet_data[subnet]["rtts_type_1_or_3"].append(rtt)

            # 计算 IID 类型并存储
            ipv6_address = ipaddress.IPv6Address(row['saddr'])
            subnet_data[subnet]["total_iids"] += 1
            iid_type = check_iid_type(ipv6_address)
            subnet_data[subnet]["iid_types"][iid_type] += 1

    # 计算统计信息并写入新CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'subnet', 'count', 'variance', 'range',
            'min_diff', 'max_diff', 'variance_diff', '95th_diff', '5th_diff', 'mean_diff',
            'variance_diff_p', 'max_diff_p', 'min_diff_p', '95th_diff_p', '5th_diff_p', 'mean_diff_p',
            'count_type_1_or_3', 'variance_type_1_or_3', 'range_type_1_or_3',
            'min_diff_type_1_or_3', 'max_diff_type_1_or_3', 'variance_diff_type_1_or_3', '95th_diff_type_1_or_3', '5th_diff_type_1_or_3', 'mean_diff_type_1_or_3',
            'variance_diff_p_type_1_or_3', 'max_diff_p_type_1_or_3', 'min_diff_p_type_1_or_3', '95th_diff_p_type_1_or_3', '5th_diff_p_type_1_or_3', 'mean_diff_p_type_1_or_3',
            'iid_macderived_ratio', 'iid_really_macderived_ratio', 'iid_lowbyte_ratio', 'iid_embeddedipv4_32_ratio', 'iid_embeddedipv4_64_ratio', 'iid_pattern_bytes_ratio',
            'iid_random_ratio', 'iid_really_macderived_over_total_ratio'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for subnet, data in subnet_data.items():
            if data["count"] > 9:  # 避免除以零错误
                # 计算原始RTT的方差和范围
                variance = calculate_variance(data["rtts"])
                r = calculate_range(data["rtts"])

                # 计算每个子网中 RTT 减去最小值后的值
                min_rtt = min(data["rtts"])
                diff_rtts = [rtt - min_rtt for rtt in data["rtts"]]

                # 计算新的RTT值的统计特征
                min_diff = find_second_smallest(diff_rtts)
                max_diff = max(diff_rtts)
                variance_diff = calculate_variance(diff_rtts)
                mean_diff = sum(diff_rtts) / len(diff_rtts)
                percentile_95 = calculate_percentile(diff_rtts, 95)
                percentile_5 = calculate_percentile(diff_rtts, 5)

                # 计算R的百分位点
                percentiles = [calculate_percentile(diff_rtts, i) for i in range(1, 101)]

                # 计算百分位差的统计特征
                stats_diff = calculate_statistics_of_differences(percentiles)

                # 计算type为1或3时的统计特征
                if len(data["rtts_type_1_or_3"]) > 5:
                    variance_type_1_or_3 = calculate_variance(data["rtts_type_1_or_3"])
                    range_type_1_or_3 = calculate_range(data["rtts_type_1_or_3"])
                    min_rtt_type_1_or_3 = min(data["rtts_type_1_or_3"])
                    diff_rtts_type_1_or_3 = [rtt - min_rtt_type_1_or_3 for rtt in data["rtts_type_1_or_3"]]
                    min_diff_type_1_or_3 = find_second_smallest(diff_rtts_type_1_or_3)
                    max_diff_type_1_or_3 = max(diff_rtts_type_1_or_3)
                    variance_diff_type_1_or_3 = calculate_variance(diff_rtts_type_1_or_3)
                    mean_diff_type_1_or_3 = sum(diff_rtts_type_1_or_3) / len(diff_rtts_type_1_or_3)
                    percentile_95_type_1_or_3 = calculate_percentile(diff_rtts_type_1_or_3, 95)
                    percentile_5_type_1_or_3 = calculate_percentile(diff_rtts_type_1_or_3, 5)
                    percentiles_type_1_or_3 = [calculate_percentile(diff_rtts_type_1_or_3, i) for i in range(1, 101)]
                    stats_diff_type_1_or_3 = calculate_statistics_of_differences(percentiles_type_1_or_3)
                else:
                    variance_type_1_or_3 = None
                    range_type_1_or_3 = None
                    min_diff_type_1_or_3 = None
                    max_diff_type_1_or_3 = None
                    variance_diff_type_1_or_3 = None
                    mean_diff_type_1_or_3 = None
                    percentile_95_type_1_or_3 = None
                    percentile_5_type_1_or_3 = None
                    stats_diff_type_1_or_3 = {
                        'variance_diff_p': None,
                        'max_diff_p': None,
                        'min_diff_p': None,
                        '95th_diff_p': None,
                        '5th_diff_p': None,
                        'mean_diff_p': None
                    }

                # 计算 IID 类型占比
                iid_ratios = {
                    'iid_macderived_ratio': data["iid_types"]["IID_MACDERIVED"] / len(data["rtts_type_1_or_3"]) if len(data["rtts_type_1_or_3"]) > 5 else None,
                    'iid_really_macderived_ratio': data["iid_types"]["IID_Really_MACDERIVED"] / len(data["rtts_type_1_or_3"]) if len(data["rtts_type_1_or_3"])  > 5 else None,
                    'iid_lowbyte_ratio': data["iid_types"]["IID_LOWBYTE"] / data["count"]  if data["count"]  > 9 else None,
                    'iid_embeddedipv4_32_ratio': data["iid_types"]["IID_EMBEDDEDIPV4_32"] / len(data["rtts_type_1_or_3"])  if len(data["rtts_type_1_or_3"])  > 5 else None,
                    'iid_embeddedipv4_64_ratio': data["iid_types"]["IID_EMBEDDEDIPV4_64"] / len(data["rtts_type_1_or_3"])  if len(data["rtts_type_1_or_3"])  > 5 else None,
                    'iid_pattern_bytes_ratio': data["iid_types"]["IID_PATTERN_BYTES"] / len(data["rtts_type_1_or_3"])  if len(data["rtts_type_1_or_3"])  > 5 else None,
                    'iid_random_ratio': data["iid_types"]["IID_RANDOM"] / len(data["rtts_type_1_or_3"])  if len(data["rtts_type_1_or_3"])  > 5 else None,
                    'iid_really_macderived_over_total_ratio': data["iid_types"]["IID_Really_MACDERIVED"] / (data["iid_types"]["IID_MACDERIVED"] + data["iid_types"]["IID_Really_MACDERIVED"]) if (data["iid_types"]["IID_MACDERIVED"] + data["iid_types"]["IID_Really_MACDERIVED"]) > 3 else None
                }

                writer.writerow({
                    'subnet': subnet,
                    'count': data["count"],
                    'variance': variance,
                    'range': r,
                    'min_diff': min_diff,
                    'max_diff': max_diff,
                    'variance_diff': variance_diff,
                    '95th_diff': percentile_95,
                    '5th_diff': percentile_5,
                    'mean_diff': mean_diff,
                    'variance_diff_p': stats_diff['variance_diff_p'],
                    'max_diff_p': stats_diff['max_diff_p'],
                    'min_diff_p': stats_diff['min_diff_p'],
                    '95th_diff_p': stats_diff['95th_diff_p'],
                    '5th_diff_p': stats_diff['5th_diff_p'],
                    'mean_diff_p': stats_diff['mean_diff_p'],
                    'count_type_1_or_3': len(data["rtts_type_1_or_3"]),
                    'variance_type_1_or_3': variance_type_1_or_3,
                    'range_type_1_or_3': range_type_1_or_3,
                    'min_diff_type_1_or_3': min_diff_type_1_or_3,
                    'max_diff_type_1_or_3': max_diff_type_1_or_3,
                    'variance_diff_type_1_or_3': variance_diff_type_1_or_3,
                    '95th_diff_type_1_or_3': percentile_95_type_1_or_3,
                    '5th_diff_type_1_or_3': percentile_5_type_1_or_3,
                    'mean_diff_type_1_or_3': mean_diff_type_1_or_3,
                    'variance_diff_p_type_1_or_3': stats_diff_type_1_or_3['variance_diff_p'],
                    'max_diff_p_type_1_or_3': stats_diff_type_1_or_3['max_diff_p'],
                    'min_diff_p_type_1_or_3': stats_diff_type_1_or_3['min_diff_p'],
                    '95th_diff_p_type_1_or_3': stats_diff_type_1_or_3['95th_diff_p'],
                    '5th_diff_p_type_1_or_3': stats_diff_type_1_or_3['5th_diff_p'],
                    'mean_diff_p_type_1_or_3': stats_diff_type_1_or_3['mean_diff_p'],
                    'iid_macderived_ratio': iid_ratios['iid_macderived_ratio'],
                    'iid_really_macderived_ratio': iid_ratios['iid_really_macderived_ratio'],
                    'iid_lowbyte_ratio': iid_ratios['iid_lowbyte_ratio'],
                    'iid_embeddedipv4_32_ratio': iid_ratios['iid_embeddedipv4_32_ratio'],
                    'iid_embeddedipv4_64_ratio': iid_ratios['iid_embeddedipv4_64_ratio'],
                    'iid_pattern_bytes_ratio': iid_ratios['iid_pattern_bytes_ratio'],
                    'iid_random_ratio': iid_ratios['iid_random_ratio'],
                    'iid_really_macderived_over_total_ratio': iid_ratios['iid_really_macderived_over_total_ratio']
                })





def process_directory(directory_path, output_dir):
    """
    处理目录下所有的CSV文件，并将结果分别保存到不同的CSV文件中。
    :param directory_path: 包含CSV文件的目录路径
    :param output_dir: 输出文件的目录路径
    """
    # 获取目录下的所有CSV文件
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    # 处理每个CSV文件
    for csv_file in csv_files:
        input_file = os.path.join(directory_path, csv_file)
        output_file = os.path.join(output_dir, f"output_{os.path.splitext(csv_file)[0]}.csv")

        process_csv(input_file, output_file)
# 调用函数处理CSV文件
if __name__ == '__main__':
    # 示例调用
    directory_path = 'E:\\Paper\\探测实验\\模型实验\\验证数据\\1018验证\\核验'
    output_dir = 'E:\\Paper\\探测实验\\模型实验\\验证数据\\1018验证\\核验\\output'

    process_directory(directory_path, output_dir)
