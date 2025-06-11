"""
RAG 文本化器组件集合
实现多种文本化策略：模板化、紧凑格式、QA格式
"""
from typing import List, Dict, Any, Optional
import logging
from app.rag.component_factory import BaseTextualizer, component_factory

logger = logging.getLogger(__name__)

# =============================================================================
# 模板文本化器
# =============================================================================

class TemplateTextualizer(BaseTextualizer):
    """模板文本化器 - 基于预定义模板的文本生成"""
    
    def __init__(self, template_type: str = 'qa', include_properties: bool = True, 
                 max_tokens: int = 2000):
        """初始化模板文本化器"""
        self.template_type = template_type
        self.include_properties = include_properties
        self.max_tokens = max_tokens
        self.is_initialized = False
        
        # 预定义模板
        self.templates = {
            'qa': self._qa_template,
            'detailed': self._detailed_template,
            'compact': self._compact_template,
            'narrative': self._narrative_template
        }
    
    def initialize(self) -> bool:
        """初始化文本化器"""
        try:
            logger.info(f"🔄 初始化模板文本化器 ({self.template_type})...")
            
            if self.template_type not in self.templates:
                logger.error(f"❌ 未支持的模板类型: {self.template_type}")
                return False
            
            self.is_initialized = True
            logger.info("✅ 模板文本化器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模板文本化器初始化失败: {str(e)}")
            return False
    
    def textualize(self, subgraph: Dict[str, Any], query: str) -> str:
        """将子图转换为文本"""
        if not self.is_initialized:
            raise RuntimeError("文本化器未初始化")
        
        template_func = self.templates[self.template_type]
        text = template_func(subgraph, query)
        
        # 截断到最大token数
        if len(text) > self.max_tokens:
            text = text[:self.max_tokens] + "..."
        
        return text
    
    def _qa_template(self, subgraph: Dict[str, Any], query: str) -> str:
        """QA格式模板"""
        text_parts = []
        text_parts.append("=== 相关图信息 ===")
        text_parts.append(f"查询: {query}")
        text_parts.append("")
        
        # 节点信息
        text_parts.append("## 实体信息:")
        nodes = subgraph.get('nodes', [])
        
        players = [n for n in nodes if n.get('type') == 'player']
        teams = [n for n in nodes if n.get('type') == 'team']
        
        if players:
            text_parts.append("### 球员:")
            for player in players:
                name = player.get('name', '未知')
                age = player.get('age', '未知')
                if self.include_properties and age != '未知':
                    text_parts.append(f"- {name} (年龄: {age}岁)")
                else:
                    text_parts.append(f"- {name}")
        
        if teams:
            text_parts.append("### 球队:")
            for team in teams:
                name = team.get('name', '未知')
                text_parts.append(f"- {name}")
        
        # 关系信息
        edges = subgraph.get('edges', [])
        if edges:
            text_parts.append("")
            text_parts.append("## 关系信息:")
            for edge in edges:
                source = edge.get('source', '')
                target = edge.get('target', '')
                relation = edge.get('relation', '相关')
                
                # 提取实体名称
                source_name = self._extract_entity_name(source)
                target_name = self._extract_entity_name(target)
                
                if relation == 'serve':
                    text_parts.append(f"- {source_name} 效力于 {target_name}")
                else:
                    text_parts.append(f"- {source_name} {relation} {target_name}")
        
        return "\n".join(text_parts)
    
    def _detailed_template(self, subgraph: Dict[str, Any], query: str) -> str:
        """详细格式模板"""
        text_parts = []
        text_parts.append("=== 详细图结构分析 ===")
        text_parts.append(f"用户查询: {query}")
        text_parts.append(f"图算法: {subgraph.get('algorithm', '未知')}")
        text_parts.append(f"节点数量: {subgraph.get('node_count', 0)}")
        text_parts.append(f"边数量: {subgraph.get('edge_count', 0)}")
        text_parts.append("")
        
        # 详细节点信息
        nodes = subgraph.get('nodes', [])
        if nodes:
            text_parts.append("## 节点详细信息:")
            for i, node in enumerate(nodes, 1):
                text_parts.append(f"### 节点 {i}:")
                text_parts.append(f"- ID: {node.get('id', '未知')}")
                text_parts.append(f"- 类型: {node.get('type', '未知')}")
                text_parts.append(f"- 名称: {node.get('name', '未知')}")
                
                if self.include_properties:
                    if node.get('age'):
                        text_parts.append(f"- 年龄: {node.get('age')}岁")
                    if node.get('importance'):
                        text_parts.append(f"- 重要性: {node.get('importance'):.3f}")
                
                text_parts.append("")
        
        # 详细边信息
        edges = subgraph.get('edges', [])
        if edges:
            text_parts.append("## 关系详细信息:")
            for i, edge in enumerate(edges, 1):
                text_parts.append(f"### 关系 {i}:")
                text_parts.append(f"- 源节点: {self._extract_entity_name(edge.get('source', ''))}")
                text_parts.append(f"- 目标节点: {self._extract_entity_name(edge.get('target', ''))}")
                text_parts.append(f"- 关系类型: {edge.get('relation', '未知')}")
                
                if self.include_properties and edge.get('weight'):
                    text_parts.append(f"- 权重: {edge.get('weight')}")
                
                text_parts.append("")
        
        return "\n".join(text_parts)
    
    def _compact_template(self, subgraph: Dict[str, Any], query: str) -> str:
        """紧凑格式模板"""
        text_parts = []
        
        nodes = subgraph.get('nodes', [])
        edges = subgraph.get('edges', [])
        
        # 简短描述
        entity_names = []
        for node in nodes:
            name = node.get('name', '未知')
            node_type = node.get('type', '')
            if node_type == 'player':
                entity_names.append(f"球员{name}")
            elif node_type == 'team':
                entity_names.append(f"球队{name}")
            else:
                entity_names.append(name)
        
        if entity_names:
            text_parts.append(f"涉及实体: {', '.join(entity_names[:5])}")  # 最多显示5个
            
            if len(entity_names) > 5:
                text_parts.append(f"等共{len(entity_names)}个实体")
        
        # 关系描述
        if edges:
            relation_descriptions = []
            for edge in edges[:3]:  # 最多显示3个关系
                source_name = self._extract_entity_name(edge.get('source', ''))
                target_name = self._extract_entity_name(edge.get('target', ''))
                relation = edge.get('relation', '相关')
                
                if relation == 'serve':
                    relation_descriptions.append(f"{source_name}效力{target_name}")
                else:
                    relation_descriptions.append(f"{source_name}{relation}{target_name}")
            
            if relation_descriptions:
                text_parts.append(f"关系: {', '.join(relation_descriptions)}")
                
                if len(edges) > 3:
                    text_parts.append(f"等共{len(edges)}个关系")
        
        return "; ".join(text_parts)
    
    def _narrative_template(self, subgraph: Dict[str, Any], query: str) -> str:
        """叙述格式模板"""
        text_parts = []
        
        nodes = subgraph.get('nodes', [])
        edges = subgraph.get('edges', [])
        
        players = [n for n in nodes if n.get('type') == 'player']
        teams = [n for n in nodes if n.get('type') == 'team']
        
        # 构建叙述
        if players and teams:
            text_parts.append("根据图数据，我们发现以下信息：")
            
            # 描述球员
            if len(players) == 1:
                player = players[0]
                name = player.get('name', '某球员')
                age = player.get('age')
                if age:
                    text_parts.append(f"{name}是一位{age}岁的球员。")
                else:
                    text_parts.append(f"{name}是一位球员。")
            else:
                player_names = [p.get('name', '未知') for p in players[:3]]
                text_parts.append(f"涉及球员包括{', '.join(player_names)}等。")
            
            # 描述关系
            serve_relations = [e for e in edges if e.get('relation') == 'serve']
            if serve_relations:
                text_parts.append("从效力关系来看：")
                for relation in serve_relations[:3]:
                    source_name = self._extract_entity_name(relation.get('source', ''))
                    target_name = self._extract_entity_name(relation.get('target', ''))
                    text_parts.append(f"- {source_name}效力于{target_name}")
        
        elif players:
            player_names = [p.get('name', '未知') for p in players[:3]]
            text_parts.append(f"主要涉及球员：{', '.join(player_names)}")
        
        elif teams:
            team_names = [t.get('name', '未知') for t in teams[:3]]
            text_parts.append(f"主要涉及球队：{', '.join(team_names)}")
        
        return " ".join(text_parts)
    
    def _extract_entity_name(self, entity_id: str) -> str:
        """从实体ID提取名称"""
        if ':' in entity_id:
            return entity_id.split(':', 1)[1]
        return entity_id

# =============================================================================
# 紧凑文本化器
# =============================================================================

class CompactTextualizer(BaseTextualizer):
    """紧凑文本化器 - 生成简洁的文本表示"""
    
    def __init__(self, max_tokens: int = 1000, prioritize_entities: bool = True):
        """初始化紧凑文本化器"""
        self.max_tokens = max_tokens
        self.prioritize_entities = prioritize_entities
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """初始化文本化器"""
        try:
            logger.info("🔄 初始化紧凑文本化器...")
            self.is_initialized = True
            logger.info("✅ 紧凑文本化器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 紧凑文本化器初始化失败: {str(e)}")
            return False
    
    def textualize(self, subgraph: Dict[str, Any], query: str) -> str:
        """将子图转换为紧凑文本"""
        if not self.is_initialized:
            raise RuntimeError("文本化器未初始化")
        
        nodes = subgraph.get('nodes', [])
        edges = subgraph.get('edges', [])
        
        if not nodes:
            return "未找到相关图信息。"
        
        # 优先显示实体
        if self.prioritize_entities:
            entity_info = self._get_priority_entities(nodes, edges)
        else:
            entity_info = self._get_all_entities(nodes, edges)
        
        # 限制长度
        if len(entity_info) > self.max_tokens:
            entity_info = entity_info[:self.max_tokens] + "..."
        
        return entity_info
    
    def _get_priority_entities(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """获取优先实体信息"""
        parts = []
        
        # 按重要性排序节点
        sorted_nodes = sorted(nodes, 
                            key=lambda x: x.get('importance', 0), 
                            reverse=True)
        
        # 选择前5个重要节点
        important_nodes = sorted_nodes[:5]
        
        players = [n for n in important_nodes if n.get('type') == 'player']
        teams = [n for n in important_nodes if n.get('type') == 'team']
        
        if players:
            player_names = [p.get('name', '未知') for p in players]
            parts.append(f"球员: {', '.join(player_names)}")
        
        if teams:
            team_names = [t.get('name', '未知') for t in teams]
            parts.append(f"球队: {', '.join(team_names)}")
        
        # 添加关键关系
        key_relations = []
        for edge in edges[:3]:  # 前3个关系
            source = self._extract_entity_name(edge.get('source', ''))
            target = self._extract_entity_name(edge.get('target', ''))
            relation = edge.get('relation', '相关')
            
            if relation == 'serve':
                key_relations.append(f"{source}→{target}")
        
        if key_relations:
            parts.append(f"效力: {', '.join(key_relations)}")
        
        return "; ".join(parts)
    
    def _get_all_entities(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """获取所有实体信息"""
        parts = []
        
        # 实体总数
        player_count = len([n for n in nodes if n.get('type') == 'player'])
        team_count = len([n for n in nodes if n.get('type') == 'team'])
        
        if player_count > 0:
            parts.append(f"{player_count}个球员")
        if team_count > 0:
            parts.append(f"{team_count}个球队")
        
        # 关系总数
        if edges:
            relation_count = len(edges)
            parts.append(f"{relation_count}个关系")
        
        # 具体实体名称（限制数量）
        entity_names = []
        for node in nodes[:6]:  # 最多6个实体
            name = node.get('name', '未知')
            entity_names.append(name)
        
        if entity_names:
            if len(nodes) > 6:
                entity_names.append("等")
            parts.append(f"包括: {', '.join(entity_names)}")
        
        return "; ".join(parts)
    
    def _extract_entity_name(self, entity_id: str) -> str:
        """从实体ID提取名称"""
        if ':' in entity_id:
            return entity_id.split(':', 1)[1]
        return entity_id

# =============================================================================
# QA专用文本化器
# =============================================================================

class QATextualizer(BaseTextualizer):
    """QA专用文本化器 - 针对问答优化的文本生成"""
    
    def __init__(self, focus_on_query: bool = True, max_tokens: int = 1500):
        """初始化QA文本化器"""
        self.focus_on_query = focus_on_query
        self.max_tokens = max_tokens
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """初始化文本化器"""
        try:
            logger.info("🔄 初始化QA文本化器...")
            self.is_initialized = True
            logger.info("✅ QA文本化器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ QA文本化器初始化失败: {str(e)}")
            return False
    
    def textualize(self, subgraph: Dict[str, Any], query: str) -> str:
        """将子图转换为QA优化的文本"""
        if not self.is_initialized:
            raise RuntimeError("文本化器未初始化")
        
        nodes = subgraph.get('nodes', [])
        edges = subgraph.get('edges', [])
        
        if not nodes:
            return "抱歉，我没有找到相关的图信息来回答您的问题。"
        
        # 分析查询类型并生成相应文本
        if self.focus_on_query:
            text = self._generate_query_focused_text(nodes, edges, query)
        else:
            text = self._generate_comprehensive_text(nodes, edges, query)
        
        # 限制长度
        if len(text) > self.max_tokens:
            text = text[:self.max_tokens] + "..."
        
        return text
    
    def _generate_query_focused_text(self, nodes: List[Dict], edges: List[Dict], 
                                   query: str) -> str:
        """生成聚焦查询的文本"""
        text_parts = []
        
        # 查询关键词提取
        query_lower = query.lower()
        
        # 根据查询类型生成文本
        if "年龄" in query_lower or "多大" in query_lower:
            text_parts.append(self._format_age_info(nodes, query))
        
        elif "球队" in query_lower or "效力" in query_lower:
            text_parts.append(self._format_team_info(nodes, edges, query))
        
        elif "球员" in query_lower:
            text_parts.append(self._format_player_info(nodes, edges, query))
        
        else:
            # 通用格式
            text_parts.append(self._format_general_info(nodes, edges, query))
        
        return "\n".join(text_parts)
    
    def _format_age_info(self, nodes: List[Dict], query: str) -> str:
        """格式化年龄信息"""
        players_with_age = [n for n in nodes 
                           if n.get('type') == 'player' and n.get('age')]
        
        if not players_with_age:
            return "未找到球员的年龄信息。"
        
        age_info = []
        for player in players_with_age:
            name = player.get('name', '未知球员')
            age = player.get('age')
            age_info.append(f"{name}：{age}岁")
        
        return "球员年龄信息：\n" + "\n".join(age_info)
    
    def _format_team_info(self, nodes: List[Dict], edges: List[Dict], 
                         query: str) -> str:
        """格式化球队信息"""
        serve_relations = [e for e in edges if e.get('relation') == 'serve']
        
        if not serve_relations:
            teams = [n for n in nodes if n.get('type') == 'team']
            if teams:
                team_names = [t.get('name', '未知') for t in teams]
                return f"相关球队：{', '.join(team_names)}"
            return "未找到球队相关信息。"
        
        team_info = []
        for relation in serve_relations:
            player_name = self._extract_entity_name(relation.get('source', ''))
            team_name = self._extract_entity_name(relation.get('target', ''))
            team_info.append(f"{player_name} 效力于 {team_name}")
        
        return "球员效力信息：\n" + "\n".join(team_info)
    
    def _format_player_info(self, nodes: List[Dict], edges: List[Dict], 
                           query: str) -> str:
        """格式化球员信息"""
        players = [n for n in nodes if n.get('type') == 'player']
        
        if not players:
            return "未找到相关球员信息。"
        
        player_info = []
        for player in players:
            name = player.get('name', '未知球员')
            age = player.get('age')
            
            info_parts = [name]
            if age:
                info_parts.append(f"年龄{age}岁")
            
            # 查找效力球队
            player_id = player.get('id', '')
            teams = []
            for edge in edges:
                if edge.get('source') == player_id and edge.get('relation') == 'serve':
                    team_name = self._extract_entity_name(edge.get('target', ''))
                    teams.append(team_name)
            
            if teams:
                info_parts.append(f"效力于{', '.join(teams)}")
            
            player_info.append(" - ".join(info_parts))
        
        return "球员信息：\n" + "\n".join(player_info)
    
    def _format_general_info(self, nodes: List[Dict], edges: List[Dict], 
                           query: str) -> str:
        """格式化通用信息"""
        text_parts = []
        
        # 实体概览
        player_count = len([n for n in nodes if n.get('type') == 'player'])
        team_count = len([n for n in nodes if n.get('type') == 'team'])
        
        overview = []
        if player_count > 0:
            overview.append(f"{player_count}个球员")
        if team_count > 0:
            overview.append(f"{team_count}个球队")
        
        if overview:
            text_parts.append(f"找到相关信息：{', '.join(overview)}")
        
        # 关键实体
        key_entities = []
        for node in nodes[:5]:  # 前5个实体
            name = node.get('name', '未知')
            node_type = node.get('type', '')
            if node_type:
                key_entities.append(f"{name}({node_type})")
            else:
                key_entities.append(name)
        
        if key_entities:
            text_parts.append(f"主要实体：{', '.join(key_entities)}")
        
        return "\n".join(text_parts)
    
    def _generate_comprehensive_text(self, nodes: List[Dict], edges: List[Dict], 
                                   query: str) -> str:
        """生成综合文本"""
        text_parts = []
        text_parts.append(f"针对查询「{query}」，找到以下相关信息：")
        text_parts.append("")
        
        # 节点信息
        if nodes:
            players = [n for n in nodes if n.get('type') == 'player']
            teams = [n for n in nodes if n.get('type') == 'team']
            
            if players:
                player_names = [p.get('name', '未知') for p in players]
                text_parts.append(f"相关球员：{', '.join(player_names)}")
            
            if teams:
                team_names = [t.get('name', '未知') for t in teams]
                text_parts.append(f"相关球队：{', '.join(team_names)}")
        
        # 关系信息
        if edges:
            serve_relations = [e for e in edges if e.get('relation') == 'serve']
            if serve_relations:
                text_parts.append("")
                text_parts.append("效力关系：")
                for relation in serve_relations[:5]:  # 最多5个关系
                    source_name = self._extract_entity_name(relation.get('source', ''))
                    target_name = self._extract_entity_name(relation.get('target', ''))
                    text_parts.append(f"- {source_name} → {target_name}")
        
        return "\n".join(text_parts)
    
    def _extract_entity_name(self, entity_id: str) -> str:
        """从实体ID提取名称"""
        if ':' in entity_id:
            return entity_id.split(':', 1)[1]
        return entity_id

# =============================================================================
# 注册所有文本化器
# =============================================================================

def register_all_textualizers():
    """注册所有文本化器到工厂"""
    component_factory.register_textualizer('template', TemplateTextualizer)
    component_factory.register_textualizer('compact', CompactTextualizer)
    component_factory.register_textualizer('qa', QATextualizer)
    logger.info("✅ 所有文本化器已注册到组件工厂")

# 自动注册
register_all_textualizers()
