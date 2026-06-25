// PageRank 单文件实现
//  - 读 Data.txt，输出 Res.txt（Top 100，格式 NodeID Score；teleport 即 kAlpha=0.85）；
//  - 稀疏矩阵（CSR）+ 分块（按源节点块做 SpMV）；
//  - 考虑 dead-end（出度 0 均匀跳转）；spider-trap 由 teleport 项缓解；
//  - 迭代至收敛（L1 阈值）；禁止调用 networkx.pagerank 等现成 API。
// 未使用第三方数值/PageRank 库，仅 C++ 标准库。

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

// ---------- Windows 下控制台 UTF-8 提示（与算法无关，便于本地调试）----------

#ifdef _WIN32
void init_console_utf8() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
}

// 直接写入 UTF-16，避免 cerr 在部分环境下仍按本地代码页解释 UTF-8 字节
void write_stderr_utf8_line(const char* utf8) {
    if (utf8 == nullptr) {
        return;
    }
    const int nw = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    if (nw <= 0) {
        return;
    }
    std::vector<wchar_t> buf(static_cast<size_t>(nw));
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, buf.data(), nw);
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD written = 0;
    const DWORD len = static_cast<DWORD>(nw - 1);
    if (WriteConsoleW(h, buf.data(), len, &written, nullptr)) {
        WriteConsoleW(h, L"\r\n", 2, &written, nullptr);
    } else {
        std::cerr << utf8 << '\n';
    }
}
#endif

// ---------- 参数与类型（与 requirement 中 teleport、Top-100 对应）----------

constexpr double kAlpha = 0.85;        // teleport（跳转）概率；必须给出 α=0.85 的结果
constexpr double kL1Tol = 1e-10;       // 相邻两轮秩向量 L1 差小于此值则视为收敛
constexpr int kMaxIter = 1000000;      // 防止异常图永不收敛的安全上限
constexpr int kTopK = 100;             // Res.txt 输出前 K 名（ Top 100）
constexpr int kSourceBlock = 256;      // 分块大小：按源节点行块遍历 CSR，体现「分块矩阵」计算

// 题目中的 NodeID 可能不连续；排序后二分查找，得到紧凑下标 0..n-1，便于稠密向量存秩。
inline int index_of_sorted(const std::vector<int>& ids, int id) {
    const auto it = std::lower_bound(ids.begin(), ids.end(), id);
    if (it == ids.end() || *it != id) {
        return -1;
    }
    return static_cast<int>(it - ids.begin());
}

struct Edge {
    int u;  // FromNodeID
    int v;  // ToNodeID
};

// 读入 Data.txt：每行 "u v" 表示一条有向边。
bool read_edges(const std::string& path, std::vector<Edge>& edges) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open input: " << path << '\n';
#ifdef _WIN32
        write_stderr_utf8_line("用法: PageRank.exe [输入图文件路径] [输出Res路径]");
        write_stderr_utf8_line(
            "默认在当前工作目录查找 Data.txt；若在别的文件夹运行 exe，请写出 Data.txt 的相对或绝对路径。");
#else
        std::cerr << "用法: PageRank.exe [输入图文件路径] [输出Res路径]\n";
        std::cerr << "默认在当前工作目录查找 Data.txt；若在别的文件夹运行 exe，请写出 Data.txt 的相对或绝对路径。\n";
#endif
        return false;
    }
    edges.reserve(160000);
    int u = 0;
    int v = 0;
    while (in >> u >> v) {
        edges.push_back({u, v});
    }
    return !edges.empty();
}

// 建图：统计图中出现的所有顶点，按 CSR（压缩稀疏行）存邻接。
// row_ptr[i]..row_ptr[i+1]-1 为节点 i 的出边在 col_idx 中的下标范围；col_idx 存邻居的紧凑下标。
// 这是「稀疏矩阵」存储，避免构造 n×n 稠密矩阵。
void build_index_and_csr(const std::vector<Edge>& edges,
                         std::vector<int>& node_ids,
                         std::vector<int>& row_ptr,
                         std::vector<int>& col_idx,
                         std::vector<int>& outdeg) {
    std::vector<int> endpoints;
    endpoints.reserve(edges.size() * 2);
    for (const auto& e : edges) {
        endpoints.push_back(e.u);
        endpoints.push_back(e.v);
    }
    std::sort(endpoints.begin(), endpoints.end());
    endpoints.erase(std::unique(endpoints.begin(), endpoints.end()), endpoints.end());
    node_ids.swap(endpoints);

    const int n = static_cast<int>(node_ids.size());

    outdeg.assign(static_cast<size_t>(n), 0);
    for (const auto& e : edges) {
        const int i = index_of_sorted(node_ids, e.u);
        const int j = index_of_sorted(node_ids, e.v);
        (void)j;
        outdeg[static_cast<size_t>(i)] += 1;
    }

    std::vector<int> cur(static_cast<size_t>(n), 0);
    row_ptr.assign(static_cast<size_t>(n + 1), 0);
    for (int i = 0; i < n; ++i) {
        row_ptr[static_cast<size_t>(i + 1)] =
            row_ptr[static_cast<size_t>(i)] + outdeg[static_cast<size_t>(i)];
    }
    const int E = row_ptr[static_cast<size_t>(n)];
    col_idx.assign(static_cast<size_t>(E), 0);

    for (const auto& e : edges) {
        const int si = index_of_sorted(node_ids, e.u);
        const int sj = index_of_sorted(node_ids, e.v);
        const int pos = row_ptr[static_cast<size_t>(si)] + cur[static_cast<size_t>(si)];
        col_idx[static_cast<size_t>(pos)] = sj;
        cur[static_cast<size_t>(si)] += 1;
    }
}

// PageRank 幂迭代：使用随机冲浪模型 r^{new} = (1-α)/n·1 + α·S^T r^{old}。
// 其中 S 在「有出边」处按出边均匀分配；dead-end 行视为均匀跳转到所有节点（此处拆成 dead_mass 均匀摊到各节点）。
// teleport 项 (1-α)/n 同时缓解 spider-trap（陷阱内概率不会永远锁死）。
void pagerank_iterate(const std::vector<int>& row_ptr,
                      const std::vector<int>& col_idx,
                      const std::vector<int>& outdeg,
                      std::vector<double>& r,
                      std::vector<double>& r_new,
                      bool verbose) {
    const int n = static_cast<int>(outdeg.size());
    const double inv_n = 1.0 / static_cast<double>(n);
    const double teleport = (1.0 - kAlpha) * inv_n;

    std::fill(r.begin(), r.end(), inv_n);

    for (int iter = 0; iter < kMaxIter; ++iter) {
        // dead-end：出度为 0 的节点在本模型下下一步均匀分布到全部 n 个节点，贡献 α·Σ r_dead / n。
        double dead_mass = 0.0;
        for (int i = 0; i < n; ++i) {
            if (outdeg[static_cast<size_t>(i)] == 0) {
                dead_mass += r[static_cast<size_t>(i)];
            }
        }
        const double dead_share = kAlpha * dead_mass * inv_n;

        // 均匀 teleport + dead-end 均匀扩散的公共部分，后续再累加「有出边」节点的边传递。
        for (int j = 0; j < n; ++j) {
            r_new[static_cast<size_t>(j)] = teleport + dead_share;
        }

        // 稀疏矩阵–向量乘（CSR）：按源节点分块遍历，对应「分块矩阵」实现思路并改善缓存局部性。
        for (int bi = 0; bi < n; bi += kSourceBlock) {
            const int i_end = std::min(n, bi + kSourceBlock);
            for (int i = bi; i < i_end; ++i) {
                const int od = outdeg[static_cast<size_t>(i)];
                if (od == 0) {
                    continue;
                }
                const double push = kAlpha * r[static_cast<size_t>(i)] / static_cast<double>(od);
                const int p0 = row_ptr[static_cast<size_t>(i)];
                const int p1 = row_ptr[static_cast<size_t>(i + 1)];
                for (int p = p0; p < p1; ++p) {
                    const int j = col_idx[static_cast<size_t>(p)];
                    r_new[static_cast<size_t>(j)] += push;
                }
            }
        }

        // 「迭代至收敛」：用全向量 L1 差作为停机准则。
        double l1 = 0.0;
        for (int i = 0; i < n; ++i) {
            l1 += std::fabs(r_new[static_cast<size_t>(i)] - r[static_cast<size_t>(i)]);
        }
        r.swap(r_new);
        if (l1 < kL1Tol) {
            if (verbose) {
                std::cerr << "Converged at iter " << (iter + 1) << ", L1=" << l1 << '\n';
            }
            break;
        }
        if (iter + 1 == kMaxIter) {
            std::cerr << "Warning: reached max iterations, L1=" << l1 << '\n';
        }
    }
}

// 输出 requirement 规定的 Res.txt：每行「原始 NodeID Score」，共 k 行（默认 Top 100）。
void write_top_k(const std::string& path,
                 const std::vector<int>& node_ids,
                 const std::vector<double>& rank,
                 int k) {
    const int n = static_cast<int>(node_ids.size());
    k = std::min(k, n);
    std::vector<int> ord(static_cast<size_t>(n));
    std::iota(ord.begin(), ord.end(), 0);
    const auto cmp = [&](int a, int b) {
        const double ra = rank[static_cast<size_t>(a)];
        const double rb = rank[static_cast<size_t>(b)];
        if (ra != rb) {
            return ra > rb;
        }
        // 分数相同时按 NodeID 升序，便于结果稳定、复现。
        return node_ids[static_cast<size_t>(a)] < node_ids[static_cast<size_t>(b)];
    };
    std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), cmp);
    ord.resize(static_cast<size_t>(k));

    std::ofstream out(path);
    if (!out) {
        std::cerr << "Cannot open output: " << path << '\n';
        return;
    }
    out << std::fixed << std::setprecision(8);
    for (int i = 0; i < k; ++i) {
        const int idx = ord[static_cast<size_t>(i)];
        out << node_ids[static_cast<size_t>(idx)] << ' '
            << rank[static_cast<size_t>(idx)] << '\n';
    }
}

}  // namespace

// 默认读 Data.txt、写 Res.txt；可通过命令行覆盖路径。可选 -v 打印收敛信息。
int main(int argc, char** argv) {
#ifdef _WIN32
    init_console_utf8();
#endif
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    const bool env_verbose = std::getenv("PAGERANK_VERBOSE") != nullptr;
    bool cli_verbose = false;
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        const std::string flag = argv[argi];
        if (flag == "-v" || flag == "--verbose") {
            cli_verbose = true;
            ++argi;
            continue;
        }
        std::cerr << "Unknown option: " << flag << '\n';
        return 1;
    }
    const bool log_verbose = env_verbose || cli_verbose;

    std::string in_file = "Data.txt";
    std::string out_file = "Res.txt";
    if (argi < argc) {
        in_file = argv[argi++];
    }
    if (argi < argc) {
        out_file = argv[argi++];
    }

    std::vector<Edge> edges;
    if (!read_edges(in_file, edges)) {
        return 1;
    }

    std::vector<int> node_ids;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<int> outdeg;
    build_index_and_csr(edges, node_ids, row_ptr, col_idx, outdeg);
    edges.clear();
    edges.shrink_to_fit();  // 构图完成后释放边列表，降低峰值内存

    const int n = static_cast<int>(node_ids.size());
    std::vector<double> r(static_cast<size_t>(n));
    std::vector<double> r_new(static_cast<size_t>(n));

    pagerank_iterate(row_ptr, col_idx, outdeg, r, r_new, log_verbose);

    write_top_k(out_file, node_ids, r, kTopK);
    if (log_verbose) {
        std::cerr << "Wrote top " << kTopK << " to " << out_file << '\n';
    }
    return 0;
}
