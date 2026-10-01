# 05 可复现篇：git / rsync / 归档 / GitHub 推送

格式：**症状 → 根因 → 修法 → 出处**。数据集案例为 pose 训练，规则通用。

## 1. rsync 双 -e 静默失败

- 症状：备份脚本 exit 0，日志里却有 36 条「remote file in list of local sources」，文件根本没传。
- 根因：脚本自带 `E="-e ssh -o ..."` 与 rsync 内置 `-e ssh` 叠加，`-o` 被 rsync 当成 `--owner` 短选项吞掉，整行命令解析错乱，全部静默失败。
- 修法：写 rsync 脚本禁用自定义 `-e` 变量（用内置 ssh 或只写 `-e ssh` 一次）；ssh 附加选项进 `~/.ssh/config`。
- 出处：backup_final_section.sh 事故（2026-09-28），已修并复跑验证。

## 2. 备份只看 exit 0 不核对大小

- 症状：以为备份成功，实际目标目录是空的或严重缺量。
- 根因：rsync 部分失败仍可能 exit 0；日志没逐条查。
- 修法：备份后 `du -sh` 源/目标对拍；跑零增量校验（见坑 5）；日志 grep 错误行。
- 出处：同上事故的教训——「备份后必核对 du -sh 大小，不能只看 exit 0」。

## 3. macOS 与 Linux rsync stats 字段不同

- 症状：校验脚本在 Mac 上抓不到 transferred 数（Linux 上好好的）。
- 根因：Linux rsync 3.x 输出 `Number of regular files transferred`，macOS openrsync（2.6.9 兼容）输出 `Number of files transferred`（无 regular），sed 模式不匹配。
- 修法：先 `rsync --version` 确认实现；sed 两个模式都试（先 Linux 后 openrsync），见 scripts/verify_rsync_archive.sh。
- 出处：2026-10-01 校验脚本踩坑。

## 4. 长距传输中断无重试无探活

- 症状：78G 归档跑到一半 ssh 超时中断，目标目录留空壳；重跑从头再来。
- 根因：Tailscale 瞬时抖动，裸 rsync 没有 --partial/--timeout/重试。
- 修法：rsync 带 `--partial --timeout=300` + 重试循环；重试前先 `ssh 探活`（如 `ssh trainer echo ok`）；完成后零增量校验。
- 出处：2026-09-28 SECTION_A 中断（MPII 23G 成空目录缺口），2026-10-01 增量批次补齐。

## 5. 归档不做零增量校验

- 症状：宣布「备份完成」，实际有目录漏传或半传。
- 根因：一次性 rsync 后不复跑验证。
- 修法：备份完成后复跑 rsync，断言 transferred=0（scripts/verify_rsync_archive.sh）；非 0 即补传，直到收敛。
- 出处：9 批次零增量校验通过才算归档完成（2026-10-01）。

## 6. git 重命名被误读为删除

- 症状：git status 显示 7 个日志「已删除」+ 7 个新文件，以为历史丢了。
- 根因：重命名（加描述性后缀）相似度 100%，git 只是暂存区未识别；`git log --follow` 能还原完整历史。
- 修法：重命名文件后先 `git add -A` 再 `git status` 看 rename 识别；改名保留排序前缀（如 YYYY-MM-DD-）；带日期的日志正文不改写（历史凭证）。
- 出处：lab-notebook 日志重命名（保留 YYYY-MM-DD- 前缀）。

## 7. 提交信息与内容不符

- 症状：commit message 说「合并旧日志」，实际是重命名；push 后无法收回。
- 根因：写 message 时凭印象，没对着 diff 写。
- 修法：commit 前 `git diff --cached --stat` 对着写；发布前发现不符就 amend/重写（本地历史）；发布后禁止改写历史，只能补丁修正。
- 出处：78f9ca5 提交前修正。

## 8. GitHub 网络受限路径与密钥权限

- 症状：push 失败（HTTP2 framing error / 75s 超时）；`gh ssh-key add` 报 404。
- 根因：github.com:443 直连被墙（api.github.com 可达）；token 缺 `admin:public_key` 权限时 gh ssh-key add 404。
- 修法：走 ssh.github.com:443（SSH config Host 别名 + IdentityFile）；密钥用 repo 级 deploy key（`gh api repos/{owner}/{repo}/keys` 建，`repo` 权限足够）；每仓库一把专用 key；push 前测连通性；`git remote -v` 确认 remote 没残留本地路径。
- 出处：2026-10-01 三仓推送全链路。

## 9. 密钥/密码进仓库

- 症状：仓库里有 token、密码、key 文件，push 到 GitHub 等于公开。
- 根因：没在提交前做全仓扫描；机器密码写进了脚本。
- 修法：推送前 `git grep -niE 'token|password|secret|api.?key'`（排除 tokenizer/token_usage/max_tokens 噪音词）；机器密码永不进文件，sudo 用 `printf 'pw\n' | ssh host "sudo -S ..."` 管道注入；已泄露的轮换。
- 出处：三仓推送前扫描 + 训练机密码铁律。

## 10. REPRODUCIBILITY.md 路径笔误

- 症状：复现清单列了路径，别人照着跑全 404。
- 根因：清单凭记忆写，没实际验证每个路径存在。
- 修法：每个列出的路径/命令实际跑一遍验证存在；数据集的下载链接写明镜像与申请页双路径；清单本身进 git 跟着仓库走。
- 出处：harness REPRODUCIBILITY.md + 三仓 README 可复现性章节约定。
