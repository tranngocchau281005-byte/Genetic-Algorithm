
# Import các thư viện cần thiết
import numpy as np
import pandas as pd
import random # tạo số ngẫu nhiên
import time # xử lý và đo thời gian
import matplotlib.pyplot as plt
import datetime as dt # làm việc với dữ liệu ngày-giờ
from scipy.stats import norm # phân phối chuẩn dùng trong mô hình thống kê

t = 'path to your dataset.csv'
dm1 = pd.read_csv(t)

dm1['time'] = pd.to_datetime(dm1['time'])
dm1.set_index('time', inplace=True)
# go dm1.info() de kiem tra

dm2 = dm1.resample('ME').last()
#Ngay cuoi thang #thang: ME #nam: MY

cols = ['CTG', 'GAS', 'BMP', 'VND', 'SSI']
missing = set(cols) - set(dm2.columns)
if missing:
    print("Thiếu mã:", missing)
else:
    dm3 = dm2[cols].sort_index(axis=1) #sap xep theo alphabet ten cot
    
#STEP 2: calculate data tinh du lieu dau vao
log_returns = np.log(dm3 / dm3.shift(1))

#Xoa nAn
#returns = returns.dropna() 
log_returns = log_returns.dropna()

#TSSl trung binh thang 
means = log_returns.mean()
print("\nTSSL trung binh hang thang:\n", means)

# Vẽ biểu đồ TSSL trung bình
means.plot(kind='bar', figsize=(8, 4))
plt.title('TSSL trung bình hàng tháng')
plt.xlabel('Cổ phiếu')
plt.ylabel('TSSL trung bình')
plt.xticks(rotation=0) # chỉnh độ của chữ
plt.tight_layout()
plt.show()

#tinh do lech chuan tung co phieu rieng le 
sd_return = log_returns.std() 
print("\nDo lech chuan:\n", sd_return)
#Tính hiệp phương sai
cov_return = log_returns.cov()
print("\nHiep phuong sai:\n", cov_return)



# ======================
# STEP 3: ENCODING CHO DANH MỤC 5 MÃ
# ======================
def chromosome(n, verbose=False): # n gen, ko hiển thị chi tiết
    """Sinh 1 nhiễm sắc thể (vector tỷ trọng danh mục)"""
    w = np.random.rand(n) # tạo tỉ trọng ngẫu nhiên
    w = w / np.sum(w)  # chuẩn hoá thành tỷ trọng
    if verbose: 
        print("New chromosome:", np.round(w, 3)) # in ra để xem thử, làm tròn các giá trị trong mảng w đến 3 chữ số thập phân.
    return w 
 

# ======================
# STEP 4: INITIAL POPULATION
# ======================
def init_population(pop_size, n):
    """Khởi tạo quần thể gồm nhiều nhiễm sắc thể"""
    return np.array([chromosome(n) for _ in range(pop_size)])

n = len(dm3.columns)       # số cổ phiếu trong danh mục
pop_size = 30              # kích thước quần thể

population = init_population(pop_size, n)
print("\nKich thuoc quan the:", population.shape)
print("Vi du 1 ca the:", np.round(population[0], 3))


# ======================
# STEP 5: FITNESS FUNCTION
# ======================
def portfolio_perf(w):
    """Tính lợi nhuận & rủi ro danh mục"""
    mu = np.dot(w, means)
    sigma = np.sqrt(np.dot(w.T, np.dot(cov_return, w))) + 1e-8 # độ lệch chuẩn
#  T: ma trận hiệp psai
    return mu, sigma

def fitness(w, rf=0.004):
    mu, sigma = portfolio_perf(w)
    # Ràng buộc
    if mu < 0.01 or sigma**2 > 0.005:   # < 1% hoặc phương sai > 0.5%
        return -999   # phạt nặng
    if sigma == 0: 
        return -999
    return (mu - rf) / sigma  # Sharpe ratio 

# Test thử
w_test = population[0]
mu, sigma = portfolio_perf(w_test)
print("\nVi du 1 ca the:")
print("Weights:", np.round(w_test, 3))
print("Return:", round(mu, 4))
print("Sigma:", round(sigma, 4))
print("Sharpe ratio:", round(fitness(w_test), 4))

# ======================
# STEP 6: SELECTION
# ======================
def selection(pop, scores, elite_frac=0.3): # chọn 30% những cá thể tốt nhất của quần thể ban đầu
    """Chọn top danh mục tốt nhất"""
    n_elite = int(len(pop) * elite_frac)
    idx = np.argsort(scores)[-n_elite:] # những cá thể có hs Sharpe cao nhất
    elite = pop[idx] # gắn vô elite
    return elite

scores = np.array([fitness(w) for w in population])
elite_pop = selection(population, scores)
print("\nKich thuoc elite population:", elite_pop.shape) 

# ======================
# STEP 7: CROSSOVER
# ======================
def crossover(p1, p2):
    """Lai ghép số học: in bố mẹ và con"""
    alpha = np.random.rand()
    child = alpha * p1 + (1 - alpha) * p2
    child /= np.sum(child)
    return child

# ======================
# STEP 8: MUTATION
# ======================
def mutation(weights):
    """Đột biến: chọn ngẫu nhiên 2 cổ phiếu để thay đổi tỷ trọng"""
    w = weights.copy()
    i, j = np.random.choice(len(w), 2, replace=False)

    # sinh đột biến
    new_i, new_j = np.random.rand(), np.random.rand()
    
    # gán & chuẩn hoá
    w[i], w[j] = new_i, new_j
    w /= np.sum(w)
    return w

# Test thu mutation
test_child_mut  = mutation(elite_pop[0])
print("Before:", elite_pop[0])
print("After :", test_child_mut)

# ======================
# STEP 9: GA LOOP
# ======================
np.random.seed(42) #so 42 de dam bao bien dc co 
generations = 40 
population = init_population(pop_size, n)

elite_frac = 0.3
best_solution, best_score = None, -999
best_scores_history = []  # lưu Sharpe tốt nhất mỗi thế hệ
diversity_history = []
mean_scores_history = []


for gen in range(generations):
    # ===== Evaluate =====
    scores = np.array([fitness(ind) for ind in population])
    # ===== Selection (Elite) =====
    elite = selection(population, scores, elite_frac)
    n_elite = elite.shape[0]

    # ===== Adaptive probabilities =====
    # nội suy xác suất crossover/mutation
    p_c = 0.6 + (0.9 - 0.6) * (gen / generations)   # từ 0.6 → 0.9
    p_m = 1 - p_c                                   # từ 0.4 → 0.1
# p_c: xac suat cua lai ghep là 0.6 thì p_m: xac suat của dot bien là 0.4
# XÁC SUẤT CỦA DOT BIÉN và LG CỌNG LẠI BẰNG 1 
# XS ĐB LUÔN NHỎ HON XS LAI GHEP 

    # ===== New population =====
    new_pop = []
    # (1) Giữ nguyên elite (elitism)
    new_pop.extend(elite)
    # (2) Sinh cá thể mới từ crossover + mutation
    while len(new_pop) < pop_size:
        # chọn bố mẹ từ elite
        p1 = elite[np.random.randint(n_elite)]
        p2 = elite[np.random.randint(n_elite)]

        # crossover
        child = crossover(p1, p2)

        # mutation SAU crossover
        if np.random.rand() < p_m:
            child = mutation(child)

        new_pop.append(child)
# cập nhật best
    gen_best_idx = np.argmax(scores)
    gen_best_score = scores[gen_best_idx]
    best_scores_history.append(gen_best_score)

    mean_scores_history.append(scores.mean())
    diversity_history.append(scores.std())
    
    if gen_best_score > best_score:
        best_score = gen_best_score
        best_solution = population[gen_best_idx].copy()
    
    population = np.array(new_pop[:pop_size])
    
    print(f"Gen {gen+1:02d} | p_c={p_c:.2f}, p_m={p_m:.2f}, Best Sharpe={best_score:.4f}") 
    
    # ======================
# VẼ ĐỒ THỊ
# ======================
plt.figure(figsize=(8,5))
plt.plot(range(1, generations+1), best_scores_history, marker='o')
plt.title("Tiến trình GA: Sharpe ratio tốt nhất mỗi thế hệ")
plt.xlabel("Thế hệ")
plt.ylabel("Best Sharpe ratio")
plt.grid(True)
plt.show()
    
# vẽ độ đa dạng của quần thể
plt.figure(figsize=(8,5))
plt.plot(range(1, generations+1), diversity_history, marker='o')
plt.title("Độ đa dạng quần thể theo thế hệ (Std of Sharpe)")
plt.xlabel("Thế hệ")
plt.ylabel("Sharpe std")
plt.grid(True)
plt.show()

mu, sigma = portfolio_perf(best_solution)
print("\n=== Danh mục tối ưu (GA) ===")
print(pd.Series(best_solution, index=dm3.columns))
print(f"\nLợi nhuận kỳ vọng: {mu:.4f}")
print(f"Rủi ro (σ):        {sigma:.4f}")
print(f"Sharpe ratio:      {best_score:.4f}")

# Quy đổi sang năm
mu_year = (1 + mu)**12 - 1
sigma_year = sigma * np.sqrt(12)
rf_year = 0.004*12   # nếu rf=0.4%/tháng
sharpe_year = (mu_year - rf_year) / sigma_year
#Tần suất theo ngày là ngày giao dịch là 252 hoặc 250 ngày/ năm  
print("\n=== Danh mục tối ưu (GA) ===")
print(pd.Series(best_solution, index=dm3.columns))

print(f"\nLợi nhuận kỳ vọng tháng: {mu:.4f}")
print(f"Rủi ro (σ) tháng:        {sigma:.4f}")
print(f"Sharpe ratio tháng:      {best_score:.4f}")

print(f"\nLợi nhuận kỳ vọng năm:   {mu_year:.4f}")
print(f"Rủi ro (σ) năm:          {sigma_year:.4f}")
print(f"Sharpe ratio năm:        {sharpe_year:.4f}")
