"""34 CICIoT2023 classes -> 8 macro-categories. Lookup is case-insensitive: the dataset spells the class 'SqlInjection',
and an exact-case key 'SQLInjection' silently maps it to 'Benign' (this bug affected every 8-category number of the first draft)."""
import numpy as np

MACRO_ORDER = ['Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web']
FINE_TO_MACRO = {
 'BenignTraffic': 'Benign', 'DictionaryBruteForce': 'BruteForce',
 'DDoS-ACK_Fragmentation': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS',
 'DDoS-ICMP_Fragmentation': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-RSTFINFlood': 'DDoS',
 'DDoS-SYN_Flood': 'DDoS', 'DDoS-SlowLoris': 'DDoS', 'DDoS-SynonymousIP_Flood': 'DDoS',
 'DDoS-TCP_Flood': 'DDoS', 'DDoS-UDP_Flood': 'DDoS', 'DDoS-UDP_Fragmentation': 'DDoS',
 'DoS-HTTP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-UDP_Flood': 'DoS',
 'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai', 'Mirai-udpplain': 'Mirai',
 'Recon-HostDiscovery': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PingSweep': 'Recon',
 'Recon-PortScan': 'Recon', 'VulnerabilityScan': 'Recon', 'MITM-ArpSpoofing': 'Spoofing',
 'DNS_Spoofing': 'Spoofing', 'BrowserHijacking': 'Web', 'CommandInjection': 'Web',
 'SqlInjection': 'Web', 'Uploading_Attack': 'Web', 'XSS': 'Web', 'Backdoor_Malware': 'Web'}
_LC = {k.lower(): v for k, v in FINE_TO_MACRO.items()}

def macro_vector(classes):
    """fine-class index -> macro-category index. Raises on an unknown class instead of silently mapping it to Benign."""
    out = []
    for c in classes:
        if c.lower() not in _LC:
            raise KeyError(f"class {c!r} has no macro-category")
        out.append(MACRO_ORDER.index(_LC[c.lower()]))
    return np.array(out)
