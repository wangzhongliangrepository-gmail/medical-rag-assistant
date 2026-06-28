# LEARN_GIT — 把 Git 用熟（面试 + 实战）

> 目标不是背命令，而是建立**心智模型**：理解 Git 在「移动指针」和「在四个区之间搬快照」。
> 想清楚这两件事，绝大多数命令你能自己推出来，面试也能讲明白「为什么」。

---

## 0. 一句话心智模型

Git 管理的是**一连串提交（commit）快照**，每个 commit 指向它的父 commit，连成一张图（DAG）。

- **commit**：某一刻整个项目的快照 + 一个父指针（merge 有两个父）。由内容算出的 40 位 SHA 唯一标识。
- **branch（分支）**：只是一个**会移动的指针**，指向某个 commit。`git commit` 就是「新建一个 commit，并把当前分支指针往前挪」。
- **HEAD**：指向「你当前在哪」。通常 HEAD → 某分支 → 某 commit。

> 记住这句：**分支很廉价，它就是个指针。** 一旦内化，merge / rebase / reset 全都好懂了。

---

## 1. 四个区：文件在哪、命令在搬什么

```
工作区(Working Dir) ──git add──▶ 暂存区(Stage/Index) ──git commit──▶ 本地仓库(.git) ──git push──▶ 远程(remote)
      ▲                                                      │
      └──────────────  git checkout / restore  ◀────────────┘
```

| 区 | 是什么 | 谁在这 |
|----|--------|--------|
| 工作区 | 你正在编辑的真实文件 | 你改的代码 |
| 暂存区(index) | 「下次要提交的内容」的草稿快照 | `git add` 放进来的 |
| 本地仓库 | 已落库的 commit 历史 | `git commit` 写进来的 |
| 远程仓库 | GitHub 上的副本 | `git push` 推上去的 |

几乎每个 Git 命令，本质都是「**把快照从某个区搬到另一个区**」或「**移动某个指针**」。看命令时先问：它动的是哪个区/哪个指针？

---

## 2. 日常工作流（80% 时间用这些）

```bash
git status                 # 现在哪些文件改了/暂存了 —— 最常敲，随时敲
git diff                   # 工作区 vs 暂存区：还没 add 的改动
git diff --staged          # 暂存区 vs 上次提交：已 add、待 commit 的改动
git add <file>             # 把指定文件改动放进暂存区
git add -p                 # ★分块挑选：同一文件里只暂存部分改动（面试加分，体现你会做干净提交）
git commit -m "feat: 加联网检索开关"
git log --oneline --graph --all   # ★看分支拓扑，一行一个 commit，强烈建议设成别名
git push                   # 推到远程
git pull                   # 拉远程 = fetch + merge（见第 6 节，注意它会产生 merge）
```

**好的提交习惯（面试官会看你的提交历史）：**
- 一个 commit 只做一件事，能用一句话说清。
- 提交信息用「类型: 动词开头的说明」，如 `fix: 修复反思重答导致短期记忆重复记问`。
- 用 `git add -p` 把无关改动拆成不同 commit，而不是一坨 `git add .`。

---

## 3. 分支与合并（核心中的核心）

```bash
git switch -c feature/login   # 新建并切到分支（新写法，比 checkout -b 更语义化）
git switch main               # 切回主分支
git branch -d feature/login   # 删除已合并的分支
git branch -D feature/login   # 强删未合并的分支（危险，确认不要了再用）
```

### merge vs rebase —— 面试最爱问，必须讲清

假设你在 `feature` 上开发，期间 `main` 又有了新提交：

```
        A───B───C   feature
       /
  D───E───F───G     main
```

**`git merge main`（在 feature 上）**：新建一个有两个父的「合并提交」M，把两条线拧到一起。
```
        A───B───C───M   feature
       /           /
  D───E───F───────G     main
```
- 优点：**忠实保留真实历史**，不改写已有 commit，安全。
- 缺点：历史里多出一堆 merge 提交，图谱可能乱。

**`git rebase main`（在 feature 上）**：把 A B C「摘下来」，以 main 最新的 G 为新基底**逐个重放**成 A' B' C'。
```
  D───E───F───G───A'──B'──C'   feature（变成一条直线）
```
- 优点：**历史是干净的直线**，好读好 review。
- 缺点：**改写了 commit（SHA 变了）**。

> **黄金法则（必背、面试常考）：不要 rebase 已经推送到远程、别人可能基于它工作的公共分支。**
> rebase 会重写历史，别人的本地会和你的对不上，逼着大家强推、冲突满天飞。
> 安全准则：**rebase 只在自己尚未分享的本地分支上用**；公共历史用 merge。

实战常见做法：在自己的 feature 分支上 `rebase main` 让历史变直、解决冲突，**合进 main 时**再用 merge（或 PR 的 squash merge）。

---

## 4. 撤销与后悔药（高频踩坑区，务必分清）

这是面试区分「会用」和「熟练」的关键——**搞清楚每个命令动的是哪个区**。

| 我想干什么 | 命令 | 动了哪个区 |
|-----------|------|-----------|
| 取消暂存（add 错了，但保留改动） | `git restore --staged <file>` | 暂存区 → 还原成 HEAD，工作区不动 |
| 丢弃工作区改动（还没 add，不要了） | `git restore <file>` | ⚠️ 工作区改动**直接没了**，不可恢复 |
| 改上一条提交信息 / 补一个文件进去 | `git commit --amend` | 重写最后一个 commit（SHA 变） |
| 撤销「最近一次提交」但保留改动在工作区 | `git reset --soft HEAD~1` | 移动分支指针，改动退回暂存区 |
| 同上但连暂存也退（改动回工作区） | `git reset HEAD~1`（默认 mixed） | 指针后退，改动回工作区 |
| 彻底丢弃最近一次提交连同改动 | `git reset --hard HEAD~1` | ⚠️⚠️ 提交和改动**全没**，最危险 |
| 撤销某个已推送的公共提交（安全） | `git revert <commit>` | **新建一个反向提交**，不改历史 ✅ |

### reset 三种模式（一张表记牢）

`git reset` 干两件事：①移动当前分支指针 ②看 `--mode` 决定要不要顺带改暂存区/工作区。

| 模式 | 移动指针 | 暂存区 | 工作区 | 用途 |
|------|---------|--------|--------|------|
| `--soft` | ✅ | 不动（保留改动） | 不动 | 把几个提交「揉」回去重新提交 |
| `--mixed`(默认) | ✅ | 重置 | 不动 | 撤销提交+取消暂存，改动还在 |
| `--hard` | ✅ | 重置 | 重置 | ⚠️ 全部丢弃，慎用 |

### reset vs revert —— 必考

- `reset`：**移动指针、改写历史**。适合「还没推送」的本地提交。推过的别 reset。
- `revert`：**新增一个抵消提交**，历史完整保留。适合「已经推送、别人看得到」的提交——这是公共分支上唯一安全的撤销方式。

> 一句话区分：**reset 当作「这事没发生过」（改历史）；revert 当作「我犯了错，现在公开纠正」（加历史）。**

---

## 5. 后悔药的后悔药：reflog（救命神技）

`git reflog` 记录了**HEAD 每一次移动**（包括被 reset/rebase 「丢掉」的提交）。
只要提交曾经存在过、且没被 gc 清理（默认保留约 90 天），就能找回。

```bash
git reflog                       # 列出 HEAD 历史，找到你想回到的那个 SHA
git reset --hard <那个SHA>       # 跳回去，「丢掉」的提交就回来了
```

> 面试金句：**「只要 commit 过，几乎都救得回来——靠 `git reflog`。」**
> 真正不可恢复的是「从没 commit、被 `git restore` / `reset --hard` 丢掉的工作区改动」。
> 所以习惯：**动手做危险操作前，先 commit 一下当存档点。**

---

## 6. 和远程协作

```bash
git clone <url>
git remote -v                    # 看远程地址
git fetch                        # 只把远程更新拉到本地，不动你的工作区（安全）
git pull                         # = fetch + merge，会直接合并到当前分支
git pull --rebase                # = fetch + rebase，避免一堆无谓 merge 提交（很多团队默认这个）
git push                         # 推当前分支
git push -u origin feature/x     # 首次推新分支并建立追踪关系
git push --force-with-lease      # ★比 --force 安全：仅当远程没被别人动过才强推
```

- `fetch` vs `pull`：`fetch` 只下载不合并（先看 `git log origin/main` 再决定怎么合，更可控）；`pull` 直接合。
- **强推用 `--force-with-lease`，别用 `--force`**：前者会检查「远程是不是我上次看到的样子」，防止覆盖掉别人刚推的提交。这个细节最能体现「熟练」。

### Pull Request 协作流（团队标准姿势）
```bash
git switch -c feature/x main     # 从最新 main 切分支
# ... 开发、提交 ...
git push -u origin feature/x     # 推上去
# 在 GitHub 开 PR → review → 合并（squash/merge/rebase 三选一）
git switch main && git pull      # 合并后回 main 拉最新
git branch -d feature/x          # 删本地分支
```

---

## 7. 处理冲突（merge / rebase 都会遇到）

冲突发生时，Git 在文件里插入标记：
```
<<<<<<< HEAD
你这边的内容
=======
对方的内容
>>>>>>> main
```
步骤：①手动编辑文件，删掉标记、留下正确的最终内容 → ②`git add <file>` 标记已解决 → ③
- merge：`git commit` 完成合并
- rebase：`git rebase --continue` 继续重放（中途想放弃用 `git rebase --abort` 回到原样）

> 实用：`git merge --abort` / `git rebase --abort` 能让你**随时退回冲突前的状态**，不用怕。

---

## 8. 几个让你显得「熟练」的实用命令

```bash
git stash                  # 临时把工作区改动收起来（去切分支救个急），工作区变干净
git stash pop              # 把收起来的改动恢复回来
git cherry-pick <commit>   # 把另一个分支的某一个提交「单独摘」到当前分支
git bisect start           # 二分查找哪个提交引入了 bug（大型项目排查神器）
git blame <file>           # 看每一行最后是谁、哪个提交改的
git log --oneline -p <file>  # 看某个文件的逐次改动历史
git tag v1.0.0             # 打版本标签（发布常用）
git clean -nd              # 预览将删除哪些未跟踪文件（-n 只看不删，确认后再 -fd）
```

### .gitignore（基本功，别把不该提交的传上去）
- 把 `.env`、密钥、`__pycache__/`、`.idea/`、大模型缓存、`qdrant_db/` 这类**本地/敏感/可重建**的东西写进 `.gitignore`。
- ⚠️ **已经提交过的文件，写进 .gitignore 也不会停止跟踪**，要 `git rm --cached <file>` 先取消跟踪再提交。
- 密钥一旦提交过，删文件不够——历史里还在。要么换密钥作废，要么用 `git filter-repo` 清历史（见第 9 节）。

---

## 9. 危险但偶尔要用（清历史里的敏感信息）

如果不小心把 `.env`/密钥提交并推送了：
1. **第一时间作废那个密钥**（在服务商后台 revoke 重置）——这比清历史更重要、更快生效。
2. 再考虑用 `git filter-repo`（或老工具 BFG）把文件从**整个历史**抹掉，然后 `--force-with-lease` 强推。
3. 通知协作者重新 clone（历史被改写了）。

> 面试讲这个能体现安全意识：**「密钥泄露，先 revoke 再清历史，顺序不能反——历史清得再干净，泄露窗口里被抓走的密钥也已经废了。」**

---

## 10. 面试高频问答速记

**Q：merge 和 rebase 区别？什么时候用哪个？**
A：merge 保留真实历史、产生合并提交、安全；rebase 把提交重放成直线历史、好读但改写了 SHA。
自己未分享的本地分支用 rebase 理顺历史；公共分支、已推送的历史用 merge。黄金法则：不 rebase 公共分支。

**Q：reset 和 revert 区别？**
A：reset 移动指针、改写历史，适合本地未推送的提交；revert 新建一个反向提交、历史完整保留，是公共分支上安全的撤销方式。

**Q：reset 的 soft/mixed/hard？**
A：都移动分支指针；soft 保留暂存区+工作区，mixed（默认）清暂存区保留工作区，hard 连工作区一起清（最危险）。

**Q：误删了提交怎么救？**
A：`git reflog` 找回 HEAD 历史里那个 SHA，再 `git reset --hard <SHA>` 跳回去。只要 commit 过基本都救得回。

**Q：fetch 和 pull 区别？**
A：fetch 只下载远程更新不合并，可控；pull = fetch + merge（或 +rebase）直接合并到当前分支。

**Q：HEAD 是什么？**
A：指向「你当前所在位置」的指针，通常 HEAD→当前分支→某 commit。`HEAD~1` 表示上一个提交。

**Q：怎么只提交一个文件的部分改动？**
A：`git add -p` 分块（hunk）挑选要暂存的部分，做出干净、单一职责的提交。

**Q：本地分支怎么和远程对应？**
A：`git push -u origin <branch>` 建立 upstream 追踪关系，之后 `git push`/`git pull` 不用再写远程和分支名。

---

## 11. 给自己配几个别名（提速 + 显专业）

```bash
git config --global alias.st status
git config --global alias.lg "log --oneline --graph --all --decorate"
git config --global alias.last "log -1 HEAD"
git config --global pull.rebase true          # pull 默认走 rebase，少一堆无谓 merge 提交
git config --global push.autoSetupRemote true # push 新分支自动建 upstream，省 -u
```

---

## 12. 怎么「练到熟」——给你一条路径

1. **每天敲 `git status` / `git log --graph`**，时刻知道自己在四个区和分支图的哪。
2. 在本项目里**真刀真枪走一遍 PR 流程**：切分支 → 改 → 提交 → 推 → 开 PR → 合并 → 删分支。
3. **故意制造场景练后悔药**：建个练习仓库，故意 `reset --hard` 删提交，再用 `reflog` 救回来；故意造冲突再解。手生的恰恰是撤销和冲突，专门练这两块。
4. 把第 10 节的问答**用自己的话讲一遍**——能讲清「为什么」，才叫熟练。

> 核心：命令会忘，但「分支是指针、命令在四个区搬快照」这个模型不会忘。理解了模型，命令查得到、推得出、讲得清。