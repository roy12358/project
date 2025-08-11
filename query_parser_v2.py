import re
import logging
from collections import namedtuple
from typing import Union, List, Dict, Optional

# ==============================================================================
# [資料結構定義 (Data Structure Definition)]
# ==============================================================================

# `Chunk` 是一個輕量級、不可變的資料容器，用於儲存解析過程中發現的所有語意片段。
Chunk = namedtuple('Chunk', ['text', 'start', 'end', 'chunk_type', 'value', 'priority'])


class QueryParser:
    """
    QueryParser - 一個針對自然語言查詢的進階意圖解析器 (最終畢業版)。

    [核心設計哲學：生成-排序-解決 (Generate-Sort-Resolve)]
    本解析器採用一個純粹的三階段架構，其智能主要體現在「生成」的完備性、
    「排序」的確定性，以及「解決」階段中分類器的智能上。

    1. 生成 (Generate): 貪婪地生成所有基於字典和規則的候選塊。
    2. 排序 (Sort): 根據「優先級 -> 長度 -> 位置」的核心鐵律排序，解決所有衝突。
    3. 解決 (Resolve): 採用“贏家通吃”策略處理主要意圖，並將所有語意理解的複雜性
                      都委派給一個智能的 `_classify_and_add_keyword` 分類器。
    """

    def __init__(self, ckip_drivers: tuple, embedding_manager, config: dict):
        """
        [建構子]
        採用「依賴注入」(Dependency Injection) 設計模式，接收外部實例化的工具。
        
        Args:
            ckip_drivers (tuple): 包含 CKIP WS, POS, NER 驅動的元組。
            embedding_manager: 用於語意相似度計算的嵌入模型管理器。
            config (dict): 包含所有規則、字典和優先級的設定檔。
        """
        self.ws, self.pos, self.ner = ckip_drivers
        self.embed_manager = embedding_manager
        self.config = config
        self.logger = logging.getLogger(__name__)

        # [優先級系統]
        # 定義不同語意類型的優先級。數值越小，優先級越高。
        # KEYWORD_CHUNK 被賦予了較高的優先級，以確保像 "不要觀光客太多"
        # 這樣的長片語能夠在排序中勝出，並觸發其複雜的「正反意圖」解析。
        self.CHUNK_PRIORITIES = {
            'BUDGET': 0,
            'PRECISE_TIME': 1,
            'NEGATED_KEYWORD_CHUNK': 2,
            'KEYWORD_CHUNK': 2,
            'NEGATED_LOCATION': 3,
            'NEGATED_RESTAURANT_TYPE': 3,
            'FUZZY_TIME': 4,
            'LOCATION': 4,
            'RESTAURANT_TYPE': 4
        }

    @staticmethod
    def _convert_num_str_to_arabic(num_str: str, config: dict) -> str:
        # 如果輸入已經是純數字，直接返回
        if num_str.isdigit():
            return num_str
        
        num_map = config.get("chinese_to_arabic_map", {})
        s = num_str.replace('兩', '二')

        # 補齊缺少單位的情況
        s = re.sub(r'萬([一二三四五六七八九])$', r'萬\1千', s)
        s = re.sub(r'千([一二三四五六七八九])$', r'千\1百', s)
        s = re.sub(r'百([一二三四五六七八九])$', r'百\1十', s)

        # 補全開頭為"十"的數字
        if s.startswith('十'):
            s = '一' + s

        total = 0
        section = 0  # 萬位以下的累積
        temp_val = 0
        units = {'十': 10, '百': 100, '千': 1000, '萬': 10000}

        for char in s:
            if char in num_map:  # 中文數字
                temp_val = num_map[char]
            elif char.isdigit():  # 阿拉伯數字
                temp_val = int(char)
            elif char in units:
                unit = units[char]
                if unit == 10000: 
                    section = (section + temp_val) * unit
                    total += section
                    section = 0
                else:
                    if temp_val == 0:
                        temp_val = 1  # 十、百、千前面省略的情況
                    section += temp_val * unit
                temp_val = 0
            # 其他符號直接略過

        total += section + temp_val
        return str(total)


    def parse(self, query_text: str) -> dict:
        """
        [主流程 | Public API]
        執行完整的解析管線 (Pipeline)，是外部調用本類別的唯一入口。
        """
        self.logger.debug("================== [ NEW PARSE REQUEST ] ==================")
        self.logger.debug(f"Original Query: '{query_text}'")
        
        processed_text = self._preprocess_text(query_text)
        if processed_text != query_text:
            self.logger.debug(f"Preprocessed Text: '{processed_text}'")
        if not processed_text.strip(): return {}
        
        candidates = self._generate_all_candidates(processed_text)
        self.logger.debug(f"Generated {len(candidates)} candidates.")
        
        candidates.sort(key=lambda c: (c.priority, -len(c.text), c.start))
        self.logger.debug(f"Sorted candidates: {[(c.text, c.chunk_type) for c in candidates]}")
        
        result = self._resolve_conflicts_and_process(candidates, processed_text)
        
        final_result = self._finalize_result(result)
        
        self.logger.debug(f"Final Parsed Result: {final_result}")
        self.logger.debug("================== [ PARSE REQUEST END ] ==================")
        return final_result

    def _preprocess_text(self, text: str) -> str:
        """
        [輔助函式 | 預處理]
        清洗並預處理原始文本。核心任務是將與「金額」相關的、不包含複雜範圍語意的單一中文數字，
        精準地轉換為阿拉伯數字。採用一個保守的、基於上下文檢查和內部協調的策略。
        """
        amount_keywords = {'預算', '大概', '日幣', '円', '元', '塊', '左右', '上下', '以下', '以內', '以上', '起跳'}
        num_pattern = re.compile(r'[\d一二兩三四五六七八九十百千萬]+')
        temp_consumed = [False] * len(text)
        
        # 步驟 1: 優先識別並「保護」中文省略範圍表達式，防止被後續邏輯錯誤地拆分處理。
        chinese_range_pattern = re.compile(r'[一二兩三四五六七八九][、\s-]*?[一二兩三四五六七八九][萬千百]')
        for match in chinese_range_pattern.finditer(text):
            self.logger.debug(f"Protecting Chinese range expression '{match.group(0)}' from preprocessing.")
            for i in range(match.start(), match.end()): temp_consumed[i] = True

        # 步驟 2: 執行單一數字的轉換，但必須尊重被保護的範圍。
        for match in reversed(list(num_pattern.finditer(text))):
            if any(temp_consumed[i] for i in range(match.start(), match.end())): continue
            num_phrase = match.group(0)
            start, end = match.span()
            prefix = text[max(0, start - 5):start]
            suffix = text[end:end + 5]
            
            # [否決規則] 擁有最高優先級，防止誤傷。
            if re.match(r'\s*(人|位|點|個|名)', suffix):
                self.logger.debug(f"Skipping conversion of '{num_phrase}' due to non-amount suffix.")
                continue
                
            is_amount = any(kw in prefix or kw in suffix for kw in amount_keywords)
            if is_amount:
                converted_num = self._convert_num_str_to_arabic(num_phrase, self.config)
                self.logger.debug(f"Preprocessing: Converted amount phrase '{num_phrase}' to '{converted_num}'.")
                text = text[:start] + converted_num + text[end:]
        return text

    def _get_all_dict_terms(self) -> list:
        """[輔助函式] 從設定檔中載入所有字典詞彙，並按長度從長到短排序。"""
        all_terms = []
        flat_locs = set()
        location_hierarchy = self.config.get("LOCATION_HIERARCHY", {})
        for super_area, details in location_hierarchy.items():
            flat_locs.add(super_area)
            flat_locs.update(details.get("children", []))
            flat_locs.update(details.get("specific_landmark_variants", []))
        station_map = self.config.get("AREA_TO_STATION_MAP", {})
        flat_locs.update(station_map.keys())
        flat_locs.update(station_map.values())
        flat_locs.update(self.config.get("LANDMARK_KEYWORDS", []))
        location_modifiers = self.config.get("LOCATION_MODIFIER_TRIGGERS", {})
        for modifier_type in location_modifiers:
            flat_locs.update(location_modifiers.get(modifier_type, []))
        all_terms.extend([(term, 'LOCATION') for term in flat_locs])
        genres = self.config.get("RESTAURANT_TYPES", []); all_terms.extend([(term, 'RESTAURANT_TYPE') for term in set(genres)])
        keywords = self.config.get("KEYWORD_TO_TAG_MAP", {}).keys(); all_terms.extend([(term, 'KEYWORD_CHUNK') for term in set(keywords)])
        return sorted(all_terms, key=lambda x: len(x[0]), reverse=True)

    def _generate_all_candidates(self, text: str) -> list:
        """[核心演算法 | 生成] 掃描文本，生成所有事實與關係候選塊。"""
        fact_candidates = []
        fact_candidates.extend(self._find_budget_candidates(text))
        fact_candidates.extend(self._find_time_candidates(text))
        all_dict_terms = self._get_all_dict_terms()
        for term, term_type in all_dict_terms:
            priority = self.CHUNK_PRIORITIES.get(term_type, 99)
            for match in re.finditer(re.escape(term), text):
                fact_candidates.append(Chunk(text=match.group(0), start=match.start(), end=match.end(), chunk_type=term_type, value=term, priority=priority))
        self.logger.debug(f"Phase 1: Generated {len(fact_candidates)} fact candidates.")
        relation_candidates = []
        negation_patterns = self.config.get("NEGATION_PATTERNS", [])
        delimiters = self.config.get("delimiters", [])
        sorted_negations = sorted(negation_patterns, key=len, reverse=True)

        for neg_pattern in sorted_negations:
            for neg_match in re.finditer(re.escape(neg_pattern), text):
                chain = self._find_longest_target_chain(text, neg_match.end(), fact_candidates, delimiters)
                if not chain: continue
                
                for i in range(len(chain['chunks'])):
                    sub_chain_chunks = chain['chunks'][:i+1]
                    sub_chain_values = tuple(c.value for c in sub_chain_chunks)
                    first_chunk_type = sub_chain_chunks[0].chunk_type
                    if not all(c.chunk_type == first_chunk_type for c in sub_chain_chunks): continue
                    sub_chain_end_pos = sub_chain_chunks[-1].end
                    full_text = text[neg_match.start():sub_chain_end_pos]
                    self.logger.info(f"Generated negation candidate: '{full_text}' for targets {sub_chain_values}")
                    negated_type = f"NEGATED_{first_chunk_type}"
                    if negated_type in self.CHUNK_PRIORITIES:
                        relation_candidates.append(Chunk(
                            text=full_text, start=neg_match.start(), end=sub_chain_end_pos,
                            chunk_type=negated_type, value=sub_chain_values,
                            priority=self.CHUNK_PRIORITIES[negated_type]
                        ))
        return fact_candidates + relation_candidates

    def _find_longest_target_chain(self, text: str, start_pos: int, fact_candidates: list, delimiters: list) -> Optional[dict]:
        """[輔助函式] 貪婪地尋找一個由分隔符連接起來的最長目標鏈。"""
        first_target = self._find_next_target(text, start_pos, fact_candidates)
        if not first_target: return None
        chain_chunks = [first_target]; current_pos = first_target.end
        while True:
            next_match = self._find_next_target_after_delimiter(text, current_pos, fact_candidates, delimiters)
            if not next_match: break
            chain_chunks.append(next_match['chunk']); current_pos = next_match['end_pos']
        return {'chunks': chain_chunks}

    def _find_next_target(self, text: str, start_pos: int, fact_candidates: list) -> Optional[Chunk]:
        """[輔助函式] 尋找緊跟在指定位置後的第一個目標實體。"""
        max_dist = self.config.get("MAX_NEGATION_DISTANCE", 12)
        potential_targets = [c for c in fact_candidates if isinstance(c.value, str) and c.start >= start_pos and c.start - start_pos < max_dist]
        return min(potential_targets, key=lambda c: c.start) if potential_targets else None

    def _find_next_target_after_delimiter(self, text: str, start_pos: int, fact_candidates: list, delimiters: list) -> Optional[dict]:
        """[輔助函式] 尋找緊跟在分隔符後的下一個目標實體。"""
        possible_next_targets = {c.value for c in fact_candidates if isinstance(c.value, str) and c.start >= start_pos and len(c.value) > 1}
        if not possible_next_targets: return None
        delim_pattern = "|".join(re.escape(d) for d in delimiters)
        target_pattern = "|".join(re.escape(t) for t in possible_next_targets)
        peek_pattern = re.compile(fr"^(?:\s*({delim_pattern})\s*|\s+)({target_pattern})")
        match = peek_pattern.search(text[start_pos:])
        if match:
            target_value = match.group(2)
            next_chunk = next((c for c in fact_candidates if isinstance(c.value, str) and c.value == target_value and c.start >= start_pos), None)
            if next_chunk: return {'chunk': next_chunk, 'end_pos': start_pos + match.end()}
        return None

    def _find_budget_candidates(self, text: str) -> list:
        """[輔助函式 | 預算解析] 採用「分層解析與內部協調」的策略，精準提取預算資訊。"""
        candidates = []
        high_certainty_patterns = {
            'chinese_range': re.compile(r'([\d一二兩三四五六七八九])[、\s-]*?([\d一二兩三四五六七八九])(萬|千|百)'),
            'range': re.compile(r'([\d一二兩三四五六七八九十百千萬]+)\s*(?:到|至|-|～)\s*([\d一二兩三四五六七八九十百千萬]+)'),
            'max': re.compile(r'(?:不(?:要)?超過|低於|少於|不用)\s*\(?(\d+)\)?|(\d+)\s*\(?(?:元|日幣|円)?\)?\s*\(?(?:以下|以內|內)\)?'),
            'min': re.compile(r'(?:超過|高於|多於)\s*\(?(\d+)\)?|(\d+)\s*\(?(?:元|日幣|円)?\)?\s*\(?(?:以上|起跳)\)?'),
            'implicit_max': re.compile(r'(?:只有|身上有|剩)\s*\(?(\d+)\)?\s*(?:元|塊|日幣|円)?')
        }
        low_certainty_patterns = {
             'ambiguous_max': re.compile(r'(?:(?:預算|大概)|(?:元|日幣|円))(?:\s*|.{0,5}?)(\d+)\s*(?:元|塊|日幣|円|左右|上下)?')
        }
        
        temp_consumed = [False] * len(text)
        
        def process_match(match, p_type):
            if any(temp_consumed[i] for i in range(match.start(), match.end())): return
            for i in range(match.start(), match.end()): temp_consumed[i] = True
            
            value = {}
            if p_type == 'chinese_range':
                num_map = self.config.get("chinese_to_arabic_map", {})
                def to_int(s): return int(s) if s.isdigit() else num_map.get(s)
                d1_str, d2_str, unit_char = match.groups()
                d1, d2 = to_int(d1_str), to_int(d2_str)
                unit_multiplier = {'萬':10000,'千': 1000, '百': 100}.get(unit_char)
                if all((d1 is not None, d2 is not None, unit_multiplier)):
                    value = {'min': d1 * unit_multiplier, 'max': d2 * unit_multiplier}
                    self.logger.info(f"Interpreting Chinese range '{match.group(0)}' as {value}.")
            else:
                def to_num(s): return int(s) if s.isdigit() else int(self._convert_num_str_to_arabic(s, self.config))
                try:
                    nums_str = [g for g in match.groups() if g]
                    if not nums_str: return
                    nums = [to_num(n) for n in nums_str]
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert budget number in '{match.group(0)}'. Skipping.")
                    return

                if p_type == 'range': value = {'min': nums[0], 'max': nums[1]}
                elif p_type in ['max', 'ambiguous_max', 'implicit_max']:
                    self.logger.info(f"Interpreting budget '{match.group(0)}' as a max value.")
                    value = {'max': nums[0]}
                elif p_type == 'min': value = {'min': nums[0]}
            
            if value:
                candidates.append(Chunk(text=match.group(0), start=match.start(), end=match.end(), chunk_type='BUDGET', value=value, priority=self.CHUNK_PRIORITIES['BUDGET']))
        
        execution_order = ['chinese_range', 'range', 'max', 'min', 'implicit_max']
        for p_type in execution_order:
            pattern = high_certainty_patterns.get(p_type)
            if pattern:
                for match in pattern.finditer(text):
                    process_match(match, p_type)

        if not candidates:
            self.logger.debug("No high-certainty budget found. Trying low-certainty patterns.")
            for p_type, pattern in low_certainty_patterns.items():
                for match in pattern.finditer(text):
                    process_match(match, p_type)
        return candidates

    def _find_time_candidates(self, text: str) -> list:
        """[輔助函式 | 時間解析] 區分精確與模糊時間，並賦予不同優先級以解決衝突。"""
        candidates = []
        time_keywords = self.config.get('time_keywords', {})
        for time_type, keywords in time_keywords.items():
            for keyword, value in keywords.items():
                for match in re.finditer(re.escape(keyword), text):
                    candidates.append(Chunk(text=match.group(0), start=match.start(), end=match.end(), chunk_type='TIME', value={'type': time_type, 'value': value, 'raw': keyword}, priority=self.CHUNK_PRIORITIES['FUZZY_TIME']))
        
        chinese_num_hour = r'(?:[一二三四五六七八九十]|二十[一二三四]?|[一二]?十[一二三四五六七八九]?)'
        time_patterns = [
            re.compile(fr'((?:[上中下]午|早上|晚上|凌晨)?\d{{1,2}}:(?:00|30))'), 
            re.compile(fr'((?:[上中下]午|早上|晚上|凌晨)?\s*(?:\d{{1,2}}|{chinese_num_hour})\s*點半?)')
        ]
        cn_to_arabic = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '兩': 2}

        def normalize_time(time_str):
            if time_str.strip() == "一點": raise ValueError("'一點' is ambiguous and is ignored.")
            time_str = time_str.replace(" ", "")
            # 先判斷是不是 HH:MM 格式
            m = re.match(r'(?:[上中下]午|早上|晚上|凌晨)?(\d{1,2}):(\d{2})', time_str)
            if m:
                hour = int(m.group(1))
                minute = int(m.group(2))
                # 判斷時間詞是否有下午、晚上需要加12小時
                if any(x in time_str for x in ['下午', '晚上', '傍晚', '中午']) and hour < 12:
                    hour += 12
                if any(x in time_str for x in ['凌晨', '早上', '上午']) and hour == 12:
                    hour = 0
                return f"{hour:02}:{minute:02}"
            else:
                # 原本中文時間邏輯
                minute = 30 if '點半' in time_str else 0
                hour_part_match = re.search(fr'(\d{{1,2}}|{chinese_num_hour})(?=點)', time_str)
                if not hour_part_match:
                    raise ValueError("No valid hour found")
                hour_str = hour_part_match.group(1)
                if hour_str.isdigit():
                    hour = int(hour_str)
                else:
                    if hour_str.startswith('二十'):
                        hour = 20 + cn_to_arabic.get(hour_str[2:], 0)
                    elif hour_str.startswith('十'):
                        hour = 10 + cn_to_arabic.get(hour_str[1:], 0)
                    else:
                        hour = cn_to_arabic[hour_str]
                # 判斷時間詞
                if ("下午" in time_str or "晚上" in time_str) and hour < 12:
                    hour += 12
                elif "凌晨" in time_str and hour >= 12:
                    hour -= 12
                elif ("上午" in time_str or "早上" in time_str) and hour == 12:
                    hour = 0
                return f"{hour:02}:{minute:02}"

        for pattern in time_patterns:
            for match in pattern.finditer(text):
                raw_text = match.group(1)
                try:
                    standard_time = normalize_time(raw_text)
                    candidates.append(Chunk(text=raw_text, start=match.start(1), end=match.end(1), chunk_type='TIME', value={'type': 'precise_time', 'value': standard_time, 'raw': raw_text}, priority=self.CHUNK_PRIORITIES['PRECISE_TIME']))
                except Exception as e:
                    self.logger.warning(f"Failed to parse precise time '{raw_text}': {e}")
        return candidates



    def _resolve_conflicts_and_process(self, sorted_candidates: list, text: str) -> dict:
        """[核心演算法 | 解決] 採用純粹的「贏家通吃」策略，將所有複雜的語意理解任務完全委派給分類器。"""
        result = {"review_tag": set(), "image_tag": set(), "genre": set(), "location": [], "time": {}, "budget": {"min": None, "max": None, "raw_text": []}, "exclude": {"tags": set(), "genres": set(), "locations": set()}}
        consumed = [False] * (len(text) + 1)

        def mark_consumed(start, end):
            for i in range(start, end): consumed[i] = True

        for c in sorted_candidates:
            if any(consumed[i] for i in range(c.start, c.end)): continue
            self.logger.info(f"處理獲勝候選項: '{c.text}' ({c.chunk_type}, 值: {c.value})")
            mark_consumed(c.start, c.end)
            
            if c.chunk_type == 'BUDGET': self._process_budget_chunk(c, result)
            elif c.chunk_type == 'TIME': self._process_time_chunk(c, result)
            elif c.chunk_type == 'LOCATION': result['location'].append({'value': c.value})
            elif c.chunk_type == 'RESTAURANT_TYPE': result['genre'].add(c.value)
            elif c.chunk_type == 'KEYWORD_CHUNK':
                self._classify_and_add_keyword(c.value, result, is_exclusion=False)
            elif c.chunk_type.startswith("NEGATED_"):
                if c.chunk_type in ['NEGATED_LOCATION', 'NEGATED_RESTAURANT_TYPE']:
                    self._process_negated_chunk(c, result)
                elif c.chunk_type == 'NEGATED_KEYWORD_CHUNK':
                    self._classify_and_add_keyword(c.value, result, is_exclusion=True)

        remaining_text = "".join([char for i, char in enumerate(text) if not consumed[i]])
        if remaining_text.strip():
            self.logger.info(f"對未知文本執行最終語意分析: '{remaining_text}'")
            self._semantic_analysis_on_remaining(remaining_text, result)
        return result

    def _process_budget_chunk(self, chunk: Chunk, result: dict):
        """[解決階段輔助函式] 處理單個預算語意片段。"""
        raw_text, budget_value = chunk.text, chunk.value
        if 'min' in budget_value:
            current_min = result['budget']['min']; result['budget']['min'] = max(current_min, budget_value['min']) if current_min is not None else budget_value['min']
        if 'max' in budget_value:
            current_max = result['budget']['max']; result['budget']['max'] = min(current_max, budget_value['max']) if current_max is not None else budget_value['max']
        if raw_text not in result['budget']['raw_text']: result['budget']['raw_text'].append(raw_text)

    def _process_time_chunk(self, chunk: Chunk, result: dict):
        """[解決階段輔助函式] 處理單個時間語意片段，並智能合併。"""
        time_value, time_type = chunk.value, chunk.value['type']
        if time_type == 'precise_time':
            result['time']['precise_time'] = time_value
            if 'time_of_day' in result['time']: del result['time']['time_of_day']
        elif time_type == 'day_of_week': result['time']['day_of_week'] = time_value
        elif time_type == 'time_of_day' and 'precise_time' not in result['time']: result['time']['time_of_day'] = time_value

    def _process_negated_chunk(self, chunk: Chunk, result: dict):
        """[解決階段輔助函式] 處理單個否定語意片段（地點和類型）。"""
        values_to_exclude = chunk.value if isinstance(chunk.value, (list, tuple)) else (chunk.value,)
        if chunk.chunk_type == 'NEGATED_LOCATION':
            for val in values_to_exclude: result['exclude']['locations'].add(val)
        elif chunk.chunk_type == 'NEGATED_RESTAURANT_TYPE':
            for val in values_to_exclude: result['exclude']['genres'].add(val)

    def _semantic_analysis_on_remaining(self, remaining_text: str, result: dict):
        """[解決階段輔助函式] 對真正的未知文本執行最終的語意推斷。"""
        try:
            words, pos_tags = self.ws([remaining_text])[0], self.pos(self.ws([remaining_text]))[0]
            stop_words = set(self.config.get("semantic_stop_words", []))
            i = 0
            while i < len(words):
                word, pos = words[i], pos_tags[i]
                if pos in ['Na', 'A', 'VH', 'VA'] and len(word) > 1 and word not in stop_words:
                    if i + 1 < len(words) and pos_tags[i+1] in ['Na', 'Nc'] and words[i+1] not in stop_words:
                        combined_word = word + words[i+1]; self._classify_and_add_keyword(combined_word, result, use_embedding_fallback=True); i += 1
                    else: self._classify_and_add_keyword(word, result, use_embedding_fallback=True)
                i += 1
        except Exception as e: self.logger.error(f"CKIP分析剩餘文本失敗: {e}")

    def _classify_and_add_keyword(self, keyword_or_tuple: Union[str, tuple], result: dict, use_embedding_fallback: bool = False, is_exclusion: bool = False):
        """[輔助函式 | 智能關鍵字分類器] 理解並處理 config.json 中複雜的正反意圖映射。"""
        keywords = keyword_or_tuple if isinstance(keyword_or_tuple, (list, tuple)) else (keyword_or_tuple,)
        keyword_map = self.config.get("KEYWORD_TO_TAG_MAP", {})
        for keyword in keywords:
            mapping_info = keyword_map.get(keyword)
            source = "字典映射"
            if isinstance(mapping_info, list):
                system_tags = mapping_info
                self.logger.info(f"關鍵字 '{keyword}' 透過 {source} 直接映射到 -> {system_tags}")
                self._apply_tags(system_tags, result, is_exclusion)
            elif isinstance(mapping_info, dict):
                if "positive_inference" in mapping_info:
                    positive_tags = mapping_info["positive_inference"]
                    self.logger.info(f"從 '{keyword}' 正面推斷出標籤 -> {positive_tags}")
                    self._apply_tags(positive_tags, result, is_exclusion=False)
                if "map_to" in mapping_info:
                    direct_tags = mapping_info["map_to"]
                    self.logger.info(f"關鍵字 '{keyword}' 透過 {source} 直接映射到 -> {direct_tags}")
                    self._apply_tags(direct_tags, result, is_exclusion)
                if "negative_target" in mapping_info:
                    target_keyword = mapping_info["negative_target"]
                    target_mapping_info = keyword_map.get(target_keyword, {})
                    tags_to_exclude = target_mapping_info if isinstance(target_mapping_info, list) else target_mapping_info.get("map_to", [])
                    if tags_to_exclude:
                        self.logger.info(f"從 '{keyword}' 的否定目標 '{target_keyword}' 推斷出排除標籤 -> {tags_to_exclude}")
                        self._apply_tags(tags_to_exclude, result, is_exclusion=True)
            elif use_embedding_fallback:
                similar_tags = self.embed_manager.find_similar_tags(keyword, threshold=0.8)
                if similar_tags:
                    source = f"詞嵌入 (查詢: '{keyword}')"
                    self.logger.info(f"關鍵字 '{keyword}' 透過 {source} -> 系統標籤: {similar_tags}")
                    self._apply_tags(similar_tags, result, is_exclusion)
            elif is_exclusion:
                self.logger.warning(f"無法找到排除關鍵字的映射: '{keyword}'")
                result["exclude"]["tags"].add(keyword)
    
    def _apply_tags(self, tags: list, result: dict, is_exclusion: bool = False):
        """[輔助工具] 將一個標籤列表（肯定或排除）安全地應用到 result 物件中。"""
        system_tags_def = self.config.get("SYSTEM_TAGS", {})
        for tag in tags:
            if is_exclusion: result["exclude"]["tags"].add(tag)
            else:
                tag_info = system_tags_def.get(tag)
                if tag_info:
                    if tag_info.get("type") == "visual": result["image_tag"].add(tag)
                    else: result["review_tag"].add(tag)

    def _finalize_result(self, result: dict) -> dict:
        """[最終化] 執行所有後處理和格式化。"""
        if 'budget' in result and result['budget']: result = self._optimize_budget(result)
        if 'location' in result and result['location']: result = self._optimize_location(result)
        result = self._resolve_conflicting_tags(result)
        result = self._standardize_format(result)
        return result

    def _optimize_budget(self, result: dict) -> dict:
        """[最終化輔助] 整理預算並檢查衝突。"""
        budget = result['budget']
        min_val, max_val = budget.get('min'), budget.get('max')
        if min_val is not None and max_val is not None and min_val > max_val:
            result['budget'] = {"error": "預算衝突：最小值大於最大值", "raw_text": ", ".join(budget.get('raw_text', []))}
        else:
            if 'raw_text' in budget: result['budget']['raw_text'] = ", ".join(budget.get('raw_text', []))
        return result

    def _optimize_location(self, result: dict) -> dict:
        """[最終化輔助] 優化地點層級與推斷。"""
        final_locs = {loc['value'] for loc in result.get('location', [])}
        macro_locations = self.config.get("MACRO_LOCATIONS", {"東京"})
        found_macro = final_locs.intersection(macro_locations)
        found_specific = final_locs.difference(found_macro)
        if found_macro and found_specific:
            self.logger.info(f"移除宏觀地點 {found_macro}，因存在更具體地點 {found_specific}")
            final_locs.difference_update(found_macro)
        station_map = self.config.get("AREA_TO_STATION_MAP", {})
        location_modifiers = self.config.get("LOCATION_MODIFIER_TRIGGERS", {})
        nearby_triggers = location_modifiers.get("nearby_station", [])
        found_triggers = [t for t in nearby_triggers if t in final_locs]
        if found_triggers:
            main_area = next((loc for loc in final_locs if loc in station_map), None)
            if main_area:
                inferred = station_map[main_area]
                self.logger.info(f"車站推斷: '{main_area}' + '{found_triggers[0]}' -> '{inferred}'")
                final_locs.add(inferred)
            for t in found_triggers: final_locs.discard(t)
        location_hierarchy = self.config.get("LOCATION_HIERARCHY", {})
        to_discard = set()
        for super_area, details in location_hierarchy.items():
            if super_area in final_locs:
                has_child = any(c in final_locs for c in details.get("children", []))
                if has_child and not any(l in final_locs for l in details.get("specific_landmark_variants", [])):
                    to_discard.add(super_area)
        final_locs -= to_discard
        to_remove = set()
        for area, station in station_map.items():
            if area in final_locs and station in final_locs: to_remove.add(area)
        final_locs -= to_remove
        result['location'] = [{"value": loc} for loc in sorted(list(final_locs))]
        return result

    def _resolve_conflicting_tags(self, result: dict) -> dict:
        """[最終化輔助] 解決肯定標籤與排除標籤的衝突。"""
        if 'exclude' not in result or 'tags' not in result['exclude']: return result
        exclude_set = set(result['exclude']['tags'])
        for key in ['review_tag', 'image_tag']:
            if key in result:
                tag_set = set(result[key])
                conflicts = tag_set.intersection(exclude_set)
                if conflicts:
                    self.logger.info(f"解決標籤衝突: {conflicts}，優先排除")
                    tag_set.difference_update(conflicts)
                    result[key] = list(tag_set)
        return result

    def _standardize_format(self, result: dict) -> dict:
        """[最終化輔助] 標準化最終輸出格式。"""
        for key in ["review_tag", "image_tag", "genre"]:
            if key in result: result[key] = sorted(list(result.get(key, set())))
        if 'genre' in result and len(result['genre']) > 1 and '餐廳' in result['genre']:
            result['genre'].remove('餐廳')
        if 'exclude' in result:
            exclude_final = {}
            for key, val_set in result["exclude"].items():
                if val_set: exclude_final[key] = sorted(list(val_set))
            if exclude_final: result["exclude"] = exclude_final
            else: del result['exclude']
        return result