"""Huffman Coding compression.

Variable-length prefix coding based on symbol frequency.
Optimal for compression when combined with other techniques.
"""

import heapq
from typing import Dict, List, Tuple, Optional


class HuffmanNode:
    """Huffman tree node."""
    
    def __init__(self, char: Optional[int], freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq


def get_frequency_map(values: List[int]) -> Dict[int, int]:
    """Build frequency map from values."""
    freq_map = {}
    for val in values:
        freq_map[val] = freq_map.get(val, 0) + 1
    return freq_map


def build_huffman_tree(freq_map: Dict[int, int]) -> Optional[HuffmanNode]:
    """Build Huffman tree from frequency map."""
    if not freq_map:
        return None
    
    heap = [HuffmanNode(char, freq) for char, freq in freq_map.items()]
    heapq.heapify(heap)
    
    if len(heap) == 1:
        node = heapq.heappop(heap)
        root = HuffmanNode(None, node.freq)
        root.left = node
        return root
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        heapq.heappush(heap, parent)
    
    return heap[0]


def generate_huffman_codes(root: Optional[HuffmanNode], 
                          code: str = '', 
                          codes: Optional[Dict[int, str]] = None) -> Dict[int, str]:
    """Generate Huffman codes from tree."""
    if codes is None:
        codes = {}
    
    if root is None:
        return codes
    
    if root.char is not None:
        codes[root.char] = code if code else '0'
        return codes
    
    if root.left:
        generate_huffman_codes(root.left, code + '0', codes)
    if root.right:
        generate_huffman_codes(root.right, code + '1', codes)
    
    return codes


def huffman_compress_rle(rle_data: List[Tuple[int, int]]) -> Tuple[str, Dict[int, str], List[int], int]:
    """Compress RLE data using Huffman coding."""
    if not rle_data:
        return '', {}, [], 0
    
    values = [val for val, count in rle_data]
    counts = [count for val, count in rle_data]
    
    freq_map = get_frequency_map(values)
    root = build_huffman_tree(freq_map)
    code_map = generate_huffman_codes(root)
    
    if len(code_map) == 1:
        single_val = next(iter(code_map))
        code_map[single_val] = '0'
    
    canonical_overhead = len(code_map) * 8
    encoded_values = ''.join(code_map[val] for val in values)
    total_bits = len(encoded_values) + (len(counts) * 8) + canonical_overhead
    
    return encoded_values, code_map, counts, total_bits


def huffman_decompress(encoded_bitstring: str, 
                      code_map: Dict[int, str], 
                      num_values: int) -> List[int]:
    """Decompress Huffman encoded bitstring back to values."""
    if not encoded_bitstring or not code_map:
        return []
    
    reverse_map = {code: value for value, code in code_map.items()}
    
    decoded_values = []
    current_code = ''
    
    for bit in encoded_bitstring:
        current_code += bit
        if current_code in reverse_map:
            decoded_values.append(reverse_map[current_code])
            current_code = ''
            if len(decoded_values) >= num_values:
                break
    
    return decoded_values


def huffman_decompress_rle(encoded_bitstring: str, 
                          code_map: Dict[int, str], 
                          rle_counts: List[int]) -> List[Tuple[int, int]]:
    """Decompress Huffman encoded RLE values."""
    values = huffman_decompress(encoded_bitstring, code_map, len(rle_counts))
    rle_data = [(val, count) for val, count in zip(values, rle_counts)]
    return rle_data
