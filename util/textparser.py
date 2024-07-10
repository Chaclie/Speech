from typing import Union

_initials = [
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "z",
    "c",
    "s",
    "zh",
    "ch",
    "sh",
    "r",
    # "y",  # 元音化被i替代
    # "w",  # 元音化被u替代
]  # 声母
_finals = [
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "e",
    "ei",
    "en",
    "eng",
    "i",  # i/yi
    "ia",  # ia/ya
    "ian",  # ian/yan
    "iang",  # iang/yang
    "iao",  # iao/yao
    "ie",  # ie/ye
    "ii",  # z/c/s单独跟随的i
    "iii",  # zh/ch/sh/r单独跟随的i
    "in",  # in/yin
    "ing",  # ing/ying
    "io",  # yo
    "iong",  # iong/yong
    "iou",  # you/iu
    "ng",
    "o",
    "ong",
    "ou",
    "u",  # u/wu
    "ua",  # ua/wa
    "uai",  # uai/wai
    "uan",  # uan/wan
    "uang",  # uang/wang
    "uei",  # wei/ui
    "uen",  # wen/un
    "ueng",  # weng
    "uo",  # wo
    "v",  # v/yu
    "van",  # van/yuan
    "ve",  # ve/yue
    "vn",  # vn/yun
]  # 韵母
_rhotic = "rr"  # 儿化音
_tones = ["1", "2", "3", "4", "5"]  # 5表轻声
_pad = "<pad>"  # 长度对齐填充
_unk = "<unk>"  # 未知拼音
_silents = ["sil", "sp1"]  # 静音或停顿
valid_tokens = [_pad, _unk] + _initials + _finals + [_rhotic] + _tones + _silents


def replace_puncts(text: str) -> str:
    """中文标点替换为英文标点"""
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    text = text.replace("！", "!")
    text = text.replace("？", "?")
    text = text.replace("；", ";")
    text = text.replace("：", ":")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("（", "(")
    text = text.replace("）", ")")
    text = text.replace("【", "[")
    text = text.replace("】", "]")
    text = text.replace("《", "<")
    text = text.replace("》", ">")
    text = text.replace("、", ",")
    text = text.replace("……", "...")
    text = text.replace("…", "...")
    return text


def split_pinyin(pinyin: str) -> tuple[str, str, str, str]:
    """
    拆分单个拼音(pinyin)为声母(initial), 韵母(final), 儿化标注(rhotic), 声调(tone);
    无法识别则返回(pinyin, <unk>, <unk>, <unk>)
    """
    initial, final, rhotic, tone = "", "", "", ""
    unk_ret = (pinyin, _unk, _unk, _unk)
    # 提取声调, 默认轻声
    if len(pinyin):
        if "0" <= pinyin[-1] and pinyin[-1] <= "9":
            if pinyin[-1] in _tones:
                tone = pinyin[-1]
            pinyin = pinyin[:-1]
        if not tone:
            tone = "5"
    else:
        return unk_ret
    # 提取儿化音
    if len(pinyin) > 1 and pinyin[-1] == "r":
        rhotic = _rhotic
        pinyin = pinyin[:-1]
    if len(pinyin):
        # y元音化
        if pinyin[0] == "y":
            # ya/yan/yang/yao/ye/yo/you/yong变y为i
            # yi/yin/ying去y, yu/yuan/yue/yun去y变u为v
            if len(pinyin) > 1:
                if pinyin[1] == "i":
                    final = pinyin[1:]
                elif pinyin[1] == "u":
                    final = "v" + pinyin[2:]
                else:
                    final = "i" + pinyin[1:]
        # w元音化
        elif pinyin[0] == "w":
            # wa/wai/wan/wang/wei/wen/weng/wo变w为u, wu去w
            if len(pinyin) > 1:
                if pinyin[1] == "u":
                    final = pinyin[1:]
                else:
                    final = "u" + pinyin[1:]
        # 鼻音
        elif pinyin in ["ng", "n", "m"]:
            final = "ng"
        # 一般
        else:
            if len(pinyin) > 1 and pinyin[:2] in ["zh", "ch", "sh"]:
                initial, final = pinyin[:2], pinyin[2:]
            elif pinyin[0] in _initials:
                initial, final = pinyin[:1], pinyin[1:]
            else:
                final = pinyin
            if initial in ["z", "c", "s"] and final == "i":
                final = "ii"
            elif initial in ["zh", "ch", "sh", "r"] and final == "i":
                final = "iii"
            # 纠正j/q/x后的u为v: j/q/x+u/uan/ue/un
            elif initial in ["j", "q", "x"] and final.startswith("u"):
                final = "v" + final[1:]
            # 韵母的特殊替代
            if final == "ue":
                final = "ve"
            elif final == "iu":
                final = "iou"
            elif final == "ui":
                final = "uei"
            elif final == "un":
                final = "uen"
    # 声母/韵母是否合法
    if (initial and initial not in _initials) or (final and final not in _finals):
        return unk_ret
    return initial, final, rhotic, tone


def merge_tokens(tokens: list[str]) -> list[tuple[int, int]]:
    """
    - merge_segs: 合并区间的起止索引(左闭右开), 主要针对韵母, 儿化音, 声调三者的合并
    """
    merge_segs: list[list[int, int]] = []
    for i, token in enumerate(tokens):
        if i > 0:
            if (token == _rhotic and tokens[i - 1] in _finals) or (
                token in _tones and tokens[i - 1] == _rhotic
            ):
                merge_segs[-1][1] += 1
                continue
        merge_segs.append([i, i + 1])
    merge_segs: list[tuple[int, int]] = [(seg[0], seg[1]) for seg in merge_segs]
    return merge_segs


def pinyins_to_tokens(pinyins: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """
    将拼音序列(pinyins)转换为符号序列(tokens), 无法识别的拼音用<unk>替代
    ---
    Returns:
    - tokens: 符号序列
    - merge_segs: 合并区间的起止索引(左闭右开), 主要针对韵母, 儿化音, 声调三者的合并
    """
    tokens: list[str] = []
    merge_segs: list[tuple[int, int]] = []
    for pinyin in pinyins:
        split_res = split_pinyin(pinyin)
        if split_res[1] == _unk:
            print(f"Unknown pinyin: {split_res[0]}")
            beg_pos = len(tokens)
            tokens.append(_silents[1])
            end_pos = len(tokens)
            merge_segs.append(tuple([beg_pos, end_pos]))
        else:
            if split_res[0]:
                beg_pos = len(tokens)
                tokens.append(split_res[0])
                end_pos = len(tokens)
                merge_segs.append(tuple([beg_pos, end_pos]))
            if split_res[1]:
                beg_pos = len(tokens)
                tokens.extend([part for part in split_res[1:] if part])
                end_pos = len(tokens)
                merge_segs.append(tuple([beg_pos, end_pos]))
    return tokens, merge_segs
