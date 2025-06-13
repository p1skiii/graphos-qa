"""
实体提取器
从分词结果中提取球员和队伍实体，并在NebulaGraph中验证其存在性
专注于实体识别和验证，不处理属性推断
"""
from typing import List, Dict, Any, Optional
from app.core.schemas import QueryContext, EntityInfo
from .base_processor import BaseNLPProcessor
from .tokenizer import Token
from app.services.database.nebula_connection import NebulaGraphConnection
import logging
import logging

logger = logging.getLogger(__name__)

class EntityExtractor(BaseNLPProcessor):
    """实体提取器 - 专注于球员和队伍实体的识别与验证"""
    
    def __init__(self):
        super().__init__("entity_extractor")
        self.nebula_conn = None
        
        # NBA 30支队伍硬编码列表
        self.nba_teams = {
            # 东部联盟
            'celtics', 'nets', 'knicks', 'sixers', 'raptors',  # 大西洋分区
            'bulls', 'cavaliers', 'pistons', 'pacers', 'bucks',  # 中部分区
            'hawks', 'hornets', 'heat', 'magic', 'wizards',  # 东南分区
            
            # 西部联盟
            'nuggets', 'timberwolves', 'thunder', 'blazers', 'jazz',  # 西北分区
            'warriors', 'clippers', 'lakers', 'suns', 'kings',  # 太平洋分区
            'mavericks', 'rockets', 'grizzlies', 'pelicans', 'spurs'  # 西南分区
        }
        
    def initialize(self) -> bool:
        """初始化实体提取器和数据库连接"""
        try:
            # 初始化NebulaGraph连接
            self.nebula_conn = NebulaGraphConnection()
            if not self.nebula_conn.connect():
                logger.error("❌ NebulaGraph连接失败")
                return False
            
            self.initialized = True
            logger.info(f"✅ {self.name} 初始化成功 (已连接NebulaGraph)")
            return True
        except Exception as e:
            logger.error(f"❌ {self.name} 初始化失败: {e}")
            return False
    
    def process(self, context: QueryContext) -> QueryContext:
        """
        提取实体并填充entity_info
        
        Args:
            context: 查询上下文
            
        Returns:
            QueryContext: 填充了entity_info的上下文
        """
        # 确保连接可用
        if not self.initialized:
            if not self.initialize():
                logger.warning("EntityExtractor未初始化，跳过实体验证")
        
        self._add_trace(context, "start_extraction")
        
        try:
            # 获取tokens
            tokens = getattr(context, 'tokens', [])
            if not tokens:
                logger.warning("没有找到tokens，跳过实体提取")
                context.entity_info = EntityInfo()
                return context
            
            # 1. 从tokens提取候选实体
            candidates = self._extract_entity_candidates(tokens)
            
            # 2. NebulaGraph验证和标准化
            validated_entities = self._validate_and_standardize(candidates)
            
            # 3. 构建EntityInfo（支持多实体）
            entity_info = EntityInfo(
                players=validated_entities['players'],
                teams=validated_entities['teams'],
                target_entities=validated_entities['target_entities']  # 🆕 简化：只保留所有目标实体
            )
            
            # 填充到context
            context.entity_info = entity_info
            
            # 添加追踪信息
            self._add_trace(context, "extraction_complete", {
                "candidate_players": candidates['persons'],
                "candidate_teams": candidates['organizations'],
                "verified_players": validated_entities['players'],
                "verified_teams": validated_entities['teams'],
                "target_entities": validated_entities['target_entities']  # 🆕 简化追踪
            })
            
            logger.debug(f"🏀 实体提取完成: 球员={validated_entities['players']}, 队伍={validated_entities['teams']}, 所有目标={validated_entities['target_entities']}")
            
        except Exception as e:
            logger.error(f"❌ 实体提取失败: {e}")
            context.entity_info = EntityInfo()
            self._add_trace(context, "extraction_error", {"error": str(e)})
        
        return context
    
    def _extract_entity_candidates(self, tokens: List[Token]) -> Dict[str, List[str]]:
        """从spaCy token中提取候选实体"""
        candidates = {
            'persons': [],  # 人名候选
            'organizations': []  # 组织/队伍候选
        }
        
        # 提取PERSON和ORG实体
        for token in tokens:
            if token.ent_type == "PERSON" and token.text not in candidates['persons']:
                candidates['persons'].append(token.text)
            elif token.ent_type == "ORG" and token.text not in candidates['organizations']:
                candidates['organizations'].append(token.text)
            # 🆕 硬编码检查：NBA队伍名
            elif (token.pos == "PROPN" and 
                  token.text.lower() in self.nba_teams and
                  token.text not in candidates['organizations']):
                candidates['organizations'].append(token.text)
        
        return candidates
    
    def _validate_and_standardize(self, candidates: Dict[str, List[str]]) -> Dict[str, Any]:
        """在NebulaGraph中验证并获取标准信息"""
        validated = {
            'players': [],
            'teams': [],
            'target_entities': []  # 🆕 简化：只保留所有目标实体
        }
        
        if not self.nebula_conn:
            logger.warning("NebulaGraph连接不可用，跳过实体验证")
            # 如果没有连接，直接返回候选实体
            validated['players'] = candidates['persons']
            validated['teams'] = candidates['organizations']
            # 构建target_entities列表
            validated['target_entities'].extend(candidates['persons'])
            validated['target_entities'].extend(candidates['organizations'])
            return validated
        
        # 验证球员
        for person in candidates['persons']:
            player_info = self._query_player_in_graph(person)
            if player_info:
                standard_name = player_info['standard_name']
                validated['players'].append(standard_name)
                validated['target_entities'].append(standard_name)
                
        # 验证队伍
        for org in candidates['organizations']:
            team_info = self._query_team_in_graph(org)
            if team_info:
                standard_name = team_info['standard_name']
                validated['teams'].append(standard_name)
                validated['target_entities'].append(standard_name)
            
        return validated
    
    def _query_player_in_graph(self, player_name: str) -> Optional[Dict[str, str]]:
        """在图数据库中查询球员"""
        try:
            query = f"MATCH (v:player) WHERE v.player.name CONTAINS '{player_name}' RETURN v.player.name AS name LIMIT 1"
            result = self.nebula_conn.execute_query(query)
            
            if result and result.get('success') and result.get('row_count', 0) > 0:
                standard_name = result['rows'][0][0]
                logger.debug(f"✅ 球员验证成功: {player_name} -> {standard_name}")
                return {'standard_name': standard_name}
            else:
                logger.debug(f"❌ 球员验证失败: {player_name} (数据库中不存在)")
                return None
                
        except Exception as e:
            logger.error(f"❌ 球员验证查询失败: {player_name}, 错误: {e}")
            return None
    
    def _query_team_in_graph(self, team_name: str) -> Optional[Dict[str, str]]:
        """在图数据库中查询队伍"""
        try:
            query = f"MATCH (v:team) WHERE v.team.name CONTAINS '{team_name}' RETURN v.team.name AS name LIMIT 1"
            result = self.nebula_conn.execute_query(query)
            
            if result and result.get('success') and result.get('row_count', 0) > 0:
                standard_name = result['rows'][0][0]
                logger.debug(f"✅ 队伍验证成功: {team_name} -> {standard_name}")
                return {'standard_name': standard_name}
            else:
                logger.debug(f"❌ 队伍验证失败: {team_name} (数据库中不存在)")
                return None
                
        except Exception as e:
            logger.error(f"❌ 队伍验证查询失败: {team_name}, 错误: {e}")
            return None
    
    def get_player_info_from_database(self, player_name: str) -> Optional[Dict[str, Any]]:
        """从数据库获取球员详细信息（供其他组件使用）"""
        if not self.nebula_conn:
            return None
        
        try:
            query = f"""
            MATCH (v:player) 
            WHERE v.player.name CONTAINS '{player_name}' 
            RETURN v.player.name AS name, 
                   v.player.age AS age, 
                   v.player.height AS height,
                   v.player.weight AS weight,
                   v.player.position AS position
            LIMIT 1
            """
            
            result = self.nebula_conn.execute_query(query)
            
            if result and result.get('success') and result.get('row_count', 0) > 0:
                row = result['rows'][0]
                return {
                    'name': row[0] if len(row) > 0 else None,
                    'age': row[1] if len(row) > 1 else None,
                    'height': row[2] if len(row) > 2 else None,
                    'weight': row[3] if len(row) > 3 else None,
                    'position': row[4] if len(row) > 4 else None
                }
            
        except Exception as e:
            logger.error(f"❌ 获取球员信息失败: {player_name}, 错误: {e}")
        
        return None
    
    def close(self):
        """手动关闭NebulaGraph连接"""
        if self.nebula_conn:
            try:
                self.nebula_conn.close()
                self.nebula_conn = None
                logger.info("✅ NebulaGraph连接已关闭")
            except Exception as e:
                logger.warning(f"关闭NebulaGraph连接时出现警告: {e}")
    
    def __del__(self):
        """析构时优雅关闭连接"""
        try:
            self.close()
        except:
            pass  # 忽略析构时的错误
