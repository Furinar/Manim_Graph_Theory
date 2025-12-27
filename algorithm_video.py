from manim import *
import heapq
import numpy as np

# ===================== 全局配置 =====================
# 颜色配置
COLOR_BFS = "#4AF626"      # 亮绿色
COLOR_DFS = "#FF3333"      # 红色
COLOR_DIJKSTRA = "#FFFF00" # 黄色
COLOR_ASTAR = "#00FFFF"    # 青色
COLOR_DEFAULT = "#FFFFFF"  # 默认白色
COLOR_VISITED = "#555555"  # 已访问灰色
COLOR_EDGE_DEFAULT = "#888888"

# 调整后的节点坐标，整体左移，拉开间距防止重叠
GRAPH_NODES = {
    "S": np.array([-6.0, 0, 0]),
    "A": np.array([-3.5, 2.5, 0]),
    "B": np.array([-3.5, -2.5, 0]),
    "C": np.array([-1.0, 2.5, 0]),
    "D": np.array([-1.0, -2.5, 0]),
    "T": np.array([1.5, 0, 0])
}

# 边及权重
GRAPH_EDGES = {
    ("S", "A"): 3, ("S", "B"): 1,
    ("A", "C"): 2, ("A", "D"): 5,
    ("B", "D"): 2, ("C", "T"): 4,
    ("D", "T"): 3, ("C", "D"): 1
}

# 启发函数
def heuristic(node_name, target_name="T"):
    pos1 = GRAPH_NODES[node_name]
    pos2 = GRAPH_NODES[target_name]
    return round(np.linalg.norm(pos1 - pos2) * 0.8, 1)

# ===================== 基础组件类 =====================
class GraphNode(VGroup):
    """自定义节点组件"""
    def __init__(self, name, position, radius=0.35, color=COLOR_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        # 增加 fill_opacity 使得节点更明显，防止背景干扰
        self.circle = Circle(radius=radius, color=color, fill_opacity=0.3, stroke_width=3)
        self.label = Text(name, font_size=24, color=color).move_to(self.circle.get_center())
        # 信息文本放在节点下方，字体稍小
        self.info_text = Text("", font_size=16, color=color).next_to(self.circle, DOWN, buff=0.15)
        self.add(self.circle, self.label, self.info_text)
        self.move_to(position)

    def update_info(self, text, color=None):
        self.remove(self.info_text)
        self.info_text = Text(str(text), font_size=16, color=color if color else self.circle.get_color()).next_to(self.circle, DOWN, buff=0.15)
        self.add(self.info_text)
        return self.info_text

    def highlight(self, color, opacity=0.6):
        return self.circle.animate.set_stroke(color=color, width=5).set_fill(color=color, opacity=opacity)

class GraphEdge(VGroup):
    """自定义边组件"""
    def __init__(self, start_node, end_node, weight, color=COLOR_EDGE_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self.start_pos = start_node.circle.get_center()
        self.end_pos = end_node.circle.get_center()
        self.weight = weight
        
        # 缩短线条两端，避免直接接触节点圆圈内部
        direction = normalize(self.end_pos - self.start_pos)
        start_point = self.start_pos + direction * 0.35
        end_point = self.end_pos - direction * 0.35
        
        self.line = Line(start_point, end_point, color=color, stroke_width=3)
        
        # 计算权重标签位置
        normal = np.array([-direction[1], direction[0], 0])
        label_pos = self.line.get_center() + normal * 0.25
        
        self.weight_label = Text(str(weight), font_size=16, color=color).move_to(label_pos)
        self.weight_bg = BackgroundRectangle(self.weight_label, color=BLACK, fill_opacity=0.7, buff=0.05)
        
        self.add(self.line, self.weight_bg, self.weight_label)

    def highlight(self, color, width=6):
        return self.line.animate.set_stroke(color=color, width=width)

class DataStructureVisualizer(VGroup):
    """数据结构可视化组件"""
    def __init__(self, type_name, color, title_text=None, **kwargs):
        super().__init__(**kwargs)
        self.type_name = type_name
        self.color = color
        self.elements = [] 
        
        # 标题
        title_str = title_text if title_text else type_name.upper()
        self.title = Text(title_str, font_size=32, color=color).to_edge(UP, buff=1.0).shift(RIGHT * 4.5)
        
        # 容器背景框 - 移到更右侧
        self.box = Rectangle(width=3.5, height=5.5, color=color, fill_opacity=0.1, stroke_width=2)
        self.box.next_to(self.title, DOWN, buff=0.2)
        
        self.container_group = VGroup()
        self.add(self.title, self.box, self.container_group)
        
        # 元素起始位置 (顶部)
        self.top_pos = self.box.get_top() + DOWN * 0.6

    def get_target_pos(self, index):
        return self.top_pos + DOWN * (index * 0.7)

    def create_element_mobject(self, element_name, value=""):
        elem_box = Rectangle(width=2.8, height=0.6, color=self.color, fill_opacity=0.4, stroke_width=1)
        text_content = f"{element_name} : {value}" if value else f"{element_name}"
        elem_text = Text(text_content, font_size=22, color=WHITE).move_to(elem_box.get_center())
        return VGroup(elem_box, elem_text)

# ===================== 视频场景 =====================
class AlgorithmComparisonVideo(Scene):
    def construct(self):
        self.camera.background_color = "#1e1e1e"
        
        self.play_opening()
        self.play_bfs_demo()
        self.play_dfs_demo()
        self.play_dijkstra_demo()
        self.play_astar_demo()
        self.play_comparison()
        self.play_summary()
        self.play_closing()

    def create_graph_visuals(self):
        nodes = {}
        edges = {}
        graph_group = VGroup()

        for name, pos in GRAPH_NODES.items():
            node = GraphNode(name, position=pos)
            nodes[name] = node
            graph_group.add(node)

        for (start, end), weight in GRAPH_EDGES.items():
            edge = GraphEdge(nodes[start], nodes[end], weight)
            edges[(start, end)] = edge
            edges[(end, start)] = edge
            graph_group.add(edge, edge.weight_label)
            
        for node in nodes.values():
            graph_group.add(node)
            
        return nodes, edges, graph_group

    def play_opening(self):
        title = Text("图搜索算法深度解析", font_size=56, gradient=(BLUE, PURPLE))
        subtitle = Text("BFS / DFS / Dijkstra / A*", font_size=36, color=GRAY).next_to(title, DOWN)
        author = Text("计科2403 潘意强制作", font_size=28, color=WHITE).next_to(subtitle, DOWN, buff=0.5)
        
        self.play(Write(title), FadeIn(subtitle), Write(author))
        self.wait(1)
        
        # 标题和副标题淡出，作者信息移动到左下角作为水印
        self.play(
            FadeOut(title), 
            FadeOut(subtitle),
            author.animate.to_corner(DL).scale(0.6).set_opacity(0.5),
            run_time=1.5
        )
        
        nodes, edges, graph_group = self.create_graph_visuals()
        info_text = Text("带权连通图 (S: 起点, T: 终点)", font_size=28, color=GRAY).to_edge(UP)
        
        self.play(Create(graph_group), run_time=2)
        self.play(FadeIn(info_text))
        self.wait(1)
        
        self.play(
            nodes["S"].highlight(GREEN),
            nodes["T"].highlight(RED),
            run_time=1
        )
        self.wait(1)
        self.play(FadeOut(info_text), FadeOut(graph_group))

    def animate_algorithm_intro(self, name, desc, color):
        # 1. 居中展示大标题
        title = Text(name, font_size=60, color=color).move_to(ORIGIN)
        description = Text(desc, font_size=32, color=GRAY).next_to(title, DOWN)
        
        self.play(Write(title), FadeIn(description))
        self.wait(1.5)
        
        # 2. 缩小并移动到左上角
        target_title = Text(name, font_size=32, color=color).to_corner(UL)
        # 描述文字放在标题右侧
        target_desc = Text(desc, font_size=20, color=GRAY).next_to(target_title, RIGHT, buff=0.5).align_to(target_title, DOWN)
        
        self.play(
            Transform(title, target_title),
            Transform(description, target_desc),
            run_time=1
        )
        return title, description

    def play_bfs_demo(self):
        title, desc = self.animate_algorithm_intro(
            "BFS: 广度优先搜索", 
            "使用队列 (Queue) - 先进先出", 
            COLOR_BFS
        )
        
        nodes, edges, graph_group = self.create_graph_visuals()
        ds_viz = DataStructureVisualizer("Queue", COLOR_BFS)
        
        self.play(FadeIn(graph_group), FadeIn(ds_viz))
        
        queue = ["S"]
        visited = {"S"}
        parent = {}
        
        # 初始入队
        elem = ds_viz.create_element_mobject("S")
        target_pos = ds_viz.get_target_pos(0)
        # 动画：从底部划入
        elem.move_to(ds_viz.box.get_bottom())
        self.play(
            nodes["S"].highlight(COLOR_BFS),
            elem.animate.move_to(target_pos),
            run_time=1
        )
        ds_viz.elements.append(elem)
        ds_viz.container_group.add(elem)
        
        found = False
        while queue:
            current = queue.pop(0)
            
            # 出队动画：向上移动并消失
            removed_elem = ds_viz.elements.pop(0)
            ds_viz.container_group.remove(removed_elem)
            
            # 剩余元素上移
            anims = [
                removed_elem.animate.shift(UP * 0.8).set_opacity(0),
            ]
            for i, el in enumerate(ds_viz.elements):
                anims.append(el.animate.move_to(ds_viz.get_target_pos(i)))
            
            self.play(*anims, run_time=0.8)
            self.remove(removed_elem)
            
            self.play(nodes[current].circle.animate.set_fill(COLOR_BFS, opacity=0.8), run_time=0.5)
            
            if current == "T":
                found = True
                break
                
            neighbors = []
            for (u, v), edge in edges.items():
                if u == current:
                    neighbors.append(v)
            neighbors.sort()
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
                    
                    # 入队动画
                    new_elem = ds_viz.create_element_mobject(neighbor)
                    new_elem.move_to(ds_viz.box.get_bottom()) # 从底部开始
                    target_pos = ds_viz.get_target_pos(len(ds_viz.elements))
                    
                    self.play(
                        edges[(current, neighbor)].highlight(COLOR_BFS, width=4),
                        nodes[neighbor].highlight(COLOR_BFS, opacity=0.3),
                        new_elem.animate.move_to(target_pos),
                        run_time=1.0
                    )
                    ds_viz.elements.append(new_elem)
                    ds_viz.container_group.add(new_elem)
                    self.wait(0.3)

        if found:
            self.show_path(parent, edges, "T", "S")

        self.play(FadeOut(title), FadeOut(desc), FadeOut(graph_group), FadeOut(ds_viz))

    def play_dfs_demo(self):
        title, desc = self.animate_algorithm_intro(
            "DFS: 深度优先搜索", 
            "使用栈 (Stack) - 后进先出", 
            COLOR_DFS
        )
        
        nodes, edges, graph_group = self.create_graph_visuals()
        ds_viz = DataStructureVisualizer("Stack", COLOR_DFS)
        
        self.play(FadeIn(graph_group), FadeIn(ds_viz))
        
        stack = ["S"]
        visited = set()
        visited.add("S")
        
        elem = ds_viz.create_element_mobject("S")
        target_pos = ds_viz.get_target_pos(0)
        elem.move_to(ds_viz.box.get_bottom())
        self.play(
            nodes["S"].highlight(COLOR_DFS),
            elem.animate.move_to(target_pos),
            run_time=1
        )
        ds_viz.elements.append(elem)
        ds_viz.container_group.add(elem)
        
        parent = {}
        found = False
        
        while stack:
            current = stack.pop()
            
            # 出栈动画 (栈顶是最后一个)：向下移动并消失
            removed_elem = ds_viz.elements.pop()
            ds_viz.container_group.remove(removed_elem)
            
            self.play(
                removed_elem.animate.shift(DOWN * 0.8).set_opacity(0),
                run_time=0.8
            )
            self.remove(removed_elem)
            
            self.play(nodes[current].circle.animate.set_fill(COLOR_DFS, opacity=0.8), run_time=0.5)
            
            if current == "T":
                found = True
                break
            
            neighbors = []
            for (u, v), edge in edges.items():
                if u == current:
                    neighbors.append(v)
            neighbors.sort(reverse=True)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    stack.append(neighbor)
                    
                    new_elem = ds_viz.create_element_mobject(neighbor)
                    new_elem.move_to(ds_viz.box.get_bottom())
                    target_pos = ds_viz.get_target_pos(len(ds_viz.elements))
                    
                    self.play(
                        edges[(current, neighbor)].highlight(COLOR_DFS, width=4),
                        nodes[neighbor].highlight(COLOR_DFS, opacity=0.3),
                        new_elem.animate.move_to(target_pos),
                        run_time=1.0
                    )
                    ds_viz.elements.append(new_elem)
                    ds_viz.container_group.add(new_elem)
                    self.wait(0.3)

        if found:
            self.show_path(parent, edges, "T", "S")

        cleanup_anims = [FadeOut(title), FadeOut(desc), FadeOut(graph_group), FadeOut(ds_viz)]
        for elem in ds_viz.elements:
            cleanup_anims.append(FadeOut(elem))
        self.play(*cleanup_anims)

    def play_dijkstra_demo(self):
        title, desc = self.animate_algorithm_intro(
            "Dijkstra: 最短路径算法", 
            "优先队列 - 贪心策略", 
            COLOR_DIJKSTRA
        )
        formula = Text("dist[v] = min(dist[v], dist[u] + weight(u, v))", font_size=24, color=COLOR_DIJKSTRA).to_edge(DOWN)
        
        nodes, edges, graph_group = self.create_graph_visuals()
        ds_viz = DataStructureVisualizer("Priority Queue", COLOR_DIJKSTRA)
        
        self.play(FadeIn(graph_group), FadeIn(ds_viz), Write(formula))
        
        distances = {node: float('inf') for node in GRAPH_NODES}
        distances["S"] = 0
        pq = [(0, "S")]
        visited = set()
        parent = {}
        
        nodes["S"].update_info("d=0")
        elem = ds_viz.create_element_mobject("S", "0")
        elem.move_to(ds_viz.box.get_bottom())
        self.play(
            nodes["S"].highlight(COLOR_DIJKSTRA),
            elem.animate.move_to(ds_viz.get_target_pos(0)),
            run_time=1
        )
        ds_viz.elements.append(elem)
        ds_viz.container_group.add(elem)
        
        while pq:
            d, current = heapq.heappop(pq)
            
            # 查找并移除视觉元素
            target_index = -1
            for i, el in enumerate(ds_viz.elements):
                # 简单匹配文本内容
                if f"{current}" in el[1].text:
                    target_index = i
                    break
            
            if target_index != -1:
                removed_elem = ds_viz.elements.pop(target_index)
                ds_viz.container_group.remove(removed_elem)
                
                anims = [removed_elem.animate.move_to(nodes[current].circle.get_center()).scale(0.5).set_opacity(0)]
                for i in range(target_index, len(ds_viz.elements)):
                    anims.append(ds_viz.elements[i].animate.move_to(ds_viz.get_target_pos(i)))
                
                self.play(*anims, run_time=1.0)
                self.remove(removed_elem)
            
            if current in visited:
                continue
            visited.add(current)
            
            self.play(nodes[current].circle.animate.set_fill(COLOR_DIJKSTRA, opacity=0.8), run_time=0.5)
            
            if current == "T":
                break
                
            for (u, v), edge in edges.items():
                if u == current and v not in visited:
                    new_dist = d + edge.weight
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        parent[v] = current
                        heapq.heappush(pq, (new_dist, v))
                        
                        self.play(edges[(current, v)].highlight(COLOR_DIJKSTRA, width=4), run_time=0.5)
                        nodes[v].update_info(f"d={new_dist}")
                        
                        new_elem = ds_viz.create_element_mobject(v, str(new_dist))
                        new_elem.move_to(ds_viz.box.get_bottom())
                        
                        # 简单处理：直接添加到末尾，不模拟堆的重排动画
                        target_pos = ds_viz.get_target_pos(len(ds_viz.elements))
                        self.play(new_elem.animate.move_to(target_pos), run_time=0.8)
                        
                        ds_viz.elements.append(new_elem)
                        ds_viz.container_group.add(new_elem)
                        self.wait(0.3)

        self.show_path(parent, edges, "T", "S")
        self.play(FadeOut(title), FadeOut(desc), FadeOut(graph_group), FadeOut(ds_viz), FadeOut(formula))

    def play_astar_demo(self):
        title, desc = self.animate_algorithm_intro(
            "A*: 启发式搜索", 
            "f(n) = g(n) + h(n)", 
            COLOR_ASTAR
        )
        formula = Text("f(n) = g(n) + h(n)", font_size=24, color=COLOR_ASTAR).to_edge(DOWN)
        
        nodes, edges, graph_group = self.create_graph_visuals()
        ds_viz = DataStructureVisualizer("Open Set", COLOR_ASTAR)
        
        self.play(FadeIn(graph_group), FadeIn(ds_viz), Write(formula))
        
        g_score = {node: float('inf') for node in GRAPH_NODES}
        g_score["S"] = 0
        f_score = {node: float('inf') for node in GRAPH_NODES}
        f_score["S"] = heuristic("S")
        
        pq = [(f_score["S"], "S")]
        visited = set()
        parent = {}
        
        nodes["S"].update_info(f"f={f_score['S']}")
        elem = ds_viz.create_element_mobject("S", f"{f_score['S']}")
        elem.move_to(ds_viz.box.get_bottom())
        self.play(
            nodes["S"].highlight(COLOR_ASTAR),
            elem.animate.move_to(ds_viz.get_target_pos(0)),
            run_time=1
        )
        ds_viz.elements.append(elem)
        ds_viz.container_group.add(elem)
        
        while pq:
            f, current = heapq.heappop(pq)
            
            target_index = -1
            for i, el in enumerate(ds_viz.elements):
                if f"{current}" in el[1].text:
                    target_index = i
                    break
            if target_index != -1:
                removed_elem = ds_viz.elements.pop(target_index)
                ds_viz.container_group.remove(removed_elem)
                
                anims = [removed_elem.animate.move_to(nodes[current].circle.get_center()).scale(0.5).set_opacity(0)]
                for i in range(target_index, len(ds_viz.elements)):
                    anims.append(ds_viz.elements[i].animate.move_to(ds_viz.get_target_pos(i)))
                
                self.play(*anims, run_time=1.0)
                self.remove(removed_elem)
                
            if current in visited:
                continue
            visited.add(current)
            
            self.play(nodes[current].circle.animate.set_fill(COLOR_ASTAR, opacity=0.8), run_time=0.5)
            
            if current == "T":
                break
                
            for (u, v), edge in edges.items():
                if u == current:
                    tentative_g = g_score[current] + edge.weight
                    if tentative_g < g_score[v]:
                        parent[v] = current
                        g_score[v] = tentative_g
                        f_score[v] = g_score[v] + heuristic(v)
                        heapq.heappush(pq, (f_score[v], v))
                        
                        self.play(edges[(current, v)].highlight(COLOR_ASTAR, width=4), run_time=0.5)
                        nodes[v].update_info(f"f={f_score[v]}")
                        
                        new_elem = ds_viz.create_element_mobject(v, f"{f_score[v]}")
                        new_elem.move_to(ds_viz.box.get_bottom())
                        target_pos = ds_viz.get_target_pos(len(ds_viz.elements))
                        self.play(new_elem.animate.move_to(target_pos), run_time=0.8)
                        
                        ds_viz.elements.append(new_elem)
                        ds_viz.container_group.add(new_elem)
                        self.wait(0.3)

        self.show_path(parent, edges, "T", "S")
        
        cleanup_anims = [FadeOut(title), FadeOut(desc), FadeOut(graph_group), FadeOut(ds_viz), FadeOut(formula)]
        for elem in ds_viz.elements:
            cleanup_anims.append(FadeOut(elem))
        self.play(*cleanup_anims)

    def show_path(self, parent, edges, start, end):
        path = []
        curr = start
        if curr in parent or curr == end:
            while curr != end:
                path.append(curr)
                curr = parent[curr]
            path.append(end)
            path.reverse()
            
            path_group = VGroup()
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                line = edges[(u, v)].line.copy().set_stroke(color=WHITE, width=8)
                path_group.add(line)
            self.play(Create(path_group), run_time=2)
            self.wait(1)
            self.play(FadeOut(path_group))

    def play_comparison(self):
        title = Text("算法路径对比", font_size=48).to_edge(UP)
        self.play(Write(title))
        
        nodes, edges, graph_group = self.create_graph_visuals()
        self.play(FadeIn(graph_group))
        
        paths = {
            "BFS": (["S", "B", "D", "T"], COLOR_BFS, 0.15),
            "DFS": (["S", "A", "C", "D", "T"], COLOR_DFS, -0.15),
            "Dijkstra": (["S", "B", "D", "T"], COLOR_DIJKSTRA, 0.05),
            "A*": (["S", "B", "D", "T"], COLOR_ASTAR, -0.05)
        }
        
        legend_group = VGroup()
        all_paths_group = VGroup() # 用于收集所有路径线条
        
        for i, (name, (path, color, offset)) in enumerate(paths.items()):
            dot = Circle(radius=0.15, color=color, fill_opacity=1)
            text = Text(name, font_size=24, color=color)
            item = VGroup(dot, text).arrange(RIGHT)
            legend_group.add(item)
            
            path_lines = VGroup()
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                start_pos = nodes[u].circle.get_center()
                end_pos = nodes[v].circle.get_center()
                
                direction = normalize(end_pos - start_pos)
                normal = np.array([-direction[1], direction[0], 0])
                
                shifted_start = start_pos + normal * offset
                shifted_end = end_pos + normal * offset
                
                line = Line(shifted_start, shifted_end, color=color, stroke_width=4)
                path_lines.add(line)
            
            all_paths_group.add(path_lines)
            self.play(Create(path_lines), run_time=1.5)
            
        legend_group.arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT, buff=1)
        self.play(FadeIn(legend_group))
        self.wait(3)
        
        self.play(FadeOut(title), FadeOut(graph_group), FadeOut(legend_group), FadeOut(all_paths_group))

    def play_summary(self):
        title = Text("算法特性总结", font_size=48).to_edge(UP)
        
        # 表头和内容
        data = [
            ["BFS", "层层推进", "Queue", "无权图最短路"],
            ["DFS", "深度优先", "Stack", "连通性/遍历"],
            ["Dijkstra", "贪心(最短)", "Priority Queue", "带权图最短路"],
            ["A*", "启发式(预估)", "Priority Queue", "高效路径规划"]
        ]
        
        # 使用 Table
        t = Table(
            data,
            col_labels=[Text("算法"), Text("策略"), Text("数据结构"), Text("适用场景")],
            include_outer_lines=True,
            line_config={"stroke_width": 1, "color": WHITE}
        )
        t.scale(0.6)
        
        # 设置样式
        # 统一字体颜色
        for mob in t.get_entries():
            mob.set_color(WHITE)
        for mob in t.get_col_labels():
            mob.set_color(YELLOW) # 表头稍微区分一下
            
        self.play(Write(title), Create(t))
        self.wait(5)
        self.play(FadeOut(title), FadeOut(t))

    def play_closing(self):
        text = Text("Thanks for Watching", font_size=60, gradient=(BLUE, PURPLE))
        self.play(GrowFromCenter(text))
        self.wait(2)
        self.play(FadeOut(text))
