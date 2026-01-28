import pygame
import torch
import numpy as np
from submission import GobangModel  # 确保这里指向你保存模型定义的文件名
from utils import _index_to_position, device

# ==========================================
# 游戏界面配置
# ==========================================
BOARD_SIZE = 12
CELL_SIZE = 45
MARGIN = 40
SCREEN_SIZE = BOARD_SIZE * CELL_SIZE + MARGIN * 2

# 颜色定义
BOARD_COLOR = (220, 179, 92)
LINE_COLOR = (0, 0, 0)
BLACK_COLOR = (10, 10, 10)
WHITE_COLOR = (245, 245, 245)
LAST_MOVE_COLOR = (200, 0, 0)

class GobangGUI:
    def __init__(self, model_path):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("五子棋：挑战 10,000 轮 ResNet 大师")
        
        # 加载并初始化模型
        self.model = GobangModel(board_size=BOARD_SIZE, bound=5).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        self.reset_game()

    def reset_game(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.game_over = False
        self.winner = 0
        self.last_move = None
        
        self.current_player = 1  # 1 为黑棋（玩家），2 为白棋（AI）

    def draw_board(self):
        self.screen.fill(BOARD_COLOR)
        # 画棋盘线
        for i in range(BOARD_SIZE):
            # 横线
            pygame.draw.line(self.screen, LINE_COLOR, 
                             (MARGIN, MARGIN + i * CELL_SIZE), 
                             (MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, MARGIN + i * CELL_SIZE), 1)
            # 纵线
            pygame.draw.line(self.screen, LINE_COLOR, 
                             (MARGIN + i * CELL_SIZE, MARGIN), 
                             (MARGIN + i * CELL_SIZE, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE), 1)
        
        # 画棋子
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] != 0:
                    color = BLACK_COLOR if self.board[r, c] == 1 else WHITE_COLOR
                    pos = (MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE)
                    pygame.draw.circle(self.screen, color, pos, CELL_SIZE // 2 - 2)
                    
                    # 标记最后一手
                    if self.last_move == (r, c):
                        pygame.draw.rect(self.screen, LAST_MOVE_COLOR, 
                                        (pos[0]-5, pos[1]-5, 10, 10))

    def check_win_simple(self, r, c, player):
        # 使用简单的判定逻辑（或者你可以调用 utils 里的 check_win）
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                nr, nc = r + dr*i, c + dc*i
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr, nc] == player: count += 1
                else: break
            for i in range(1, 5):
                nr, nc = r - dr*i, c - dc*i
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr, nc] == player: count += 1
                else: break
            if count >= 5: return True
        return False

    def ai_move(self):
        state = np.copy(self.board)
        ai_state = np.where(state == 2, 1, np.where(state == 1, 2, 0))
        
        with torch.no_grad():
            # 模型输入应为 [1, 1, 12, 12]
            input_tensor = torch.tensor(ai_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            policy = self.model.actor(input_tensor)[0].cpu().numpy()
            
            # 选择概率最高且合法的点
            action_idx = np.argmax(policy)
            r, c = _index_to_position(BOARD_SIZE, action_idx)
            
            self.board[r, c] = 2
            self.last_move = (r, c)
            if self.check_win_simple(r, c, 2):
                self.game_over = True
                self.winner = 2

    def run(self):
        clock = pygame.time.Clock()
        while True:
            self.draw_board()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                
                if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over and self.current_player == 1:
                    x, y = event.pos
                    c = round((x - MARGIN) / CELL_SIZE)
                    r = round((y - MARGIN) / CELL_SIZE)
                    
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c] == 0:
                        self.board[r, c] = 1
                        self.last_move = (r, c)
                        if self.check_win_simple(r, c, 1):
                            self.game_over = True
                            self.winner = 1
                        else:
                            self.current_player = 2 # 换 AI
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: self.reset_game()

            # AI 自动行动
            if not self.game_over and self.current_player == 2:
                self.ai_move()
                self.current_player = 1

            if self.game_over:
                msg = "You Win!" if self.winner == 1 else "AI (ResNet) Wins!"
                pygame.display.set_caption(f"GAME OVER - {msg} (Press R to restart)")

            pygame.display.flip()
            clock.tick(30)

if __name__ == "__main__":
    
    MODEL_PATH = "checkpoints/model_bestbest.pth" 
    gui = GobangGUI(MODEL_PATH)
    gui.run()