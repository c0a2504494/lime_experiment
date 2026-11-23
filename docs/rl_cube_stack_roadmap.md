# TurtleBot3 Lime キューブ2段積み 強化学習ロードマップ

目標: MuJoCo上で「キューブAをキューブBの上に積む」ポリシーを強化学習し、
そのまま実機の TurtleBot3 Lime に移植して現実世界で2段積みを成功させる。

対象アーム: `open_manipulator_sara`(6自由度回転関節 + 平行グリッパ)
対象キューブ: 一辺57mm / 質量約90g(想定70〜105gでランダム化)

---

## Phase 0: 事前検証(完了済み)

- [x] `turtlebot3_lime_description` を GitHub(ROBOTIS-JAPAN-GIT, Apache-2.0)から取得
      → `external/turtlebot3_lime/turtlebot3_lime_description/`(git管理外・再取得コマンドは下記)

  ```
  git clone --depth 1 --branch humble-devel --filter=blob:none --sparse ^
    https://github.com/ROBOTIS-JAPAN-GIT/turtlebot3_lime.git external/turtlebot3_lime
  cd external/turtlebot3_lime && git sparse-checkout set turtlebot3_lime_description
  ```
- [x] グリッパ幾何の検証: 指パッド面の開口 **22〜80mm** → 57mmキューブは把持可能
- [x] 摩擦係数の方針決定: ABS樹脂 μ_slide ≒ 0.3〜0.5。
      学習では **0.3〜1.0 の範囲でドメインランダム化**(実機の不確かさを吸収)

---

## Phase 1: MuJoCoシミュレーション環境の構築

### 1-1. xacro → URDF 展開
- `pip install xacro` で `open_manipulator_sara.urdf.xacro` を展開(ROS不要)。
- 出力: `open_manipulator_sara.urdf`(メッシュ参照を絶対/相対パスで解決済みのもの)

### 1-2. URDF → MJCF 変換
- MuJoCoはURDFを直接コンパイル可能(`mjcf.fromPath` / `compiler`)。
- 対応が必要な差分:
  - **STLスケール**: URDFで `scale="0.001"` → MJCF側で `mesh scale` か `<compiler meshdir>` + asset定義
  - **mimic関節**: `gripper_right_joint` は `<mimic>` → MuJoCoでは `<equality><joint .../></equality>` に変換
  - **アクチュエータ**: URDFには無い → MJCFで `<position>` アクチュエータ(関節位置制御)と `<position>`(グリッパ)を追加。実機のDynamixelは位置制御なのでこれが最も現実に近い
  - **ジョイントダイナミクス**: `<dynamics damping friction>` は `dof damping` / `frictionloss` に反映
- シーン構成(推奨): **アームを台座に固定**(台車は使わない)。
  テーブル面 + キューブ2個 + アーム、で最小構成にする。
  ※ 台車込みだと探索空間・制御遅延が増え学習が一桁難しくなる。実機デプロイ時も
  「台車をキューブ前に駐車 → アームだけ動かす」運用で同じ結果が得られる。

### 1-3. Gymnasium環境の実装 (`rubik_stack_env.py`)
- **観測(obs)**: 関節角度6・関節速度6・EE位置姿勢・グリッパ開度・
  キューブA/Bの位置+クォータニオン+速度・EE-キューブ間相対ベクトル
- **行動(action)**: EE位置のデルタ(Δx,Δy,Δz, ±数mm/step) + グリッパ開閉1次元
  - 実装は関節位置目標にIK(数値IK or 差分ヤコビアン)で変換 → 実機移植時も同じI/Fにできる
- **報酬(shaped)**: 段階報酬
  1. EE→キューブA距離の減少
  2. 把持成功(キューブが指に接触 & 持ち上がる)
  3. キューブA高さの増加
  4. A→B真上への水平距離減少
  5. B上に載せて離す → 大きな成功ボーナス
- **終了条件**: 成功 / キューブ落下 / タイムアウト
- **成功判定**: キューブAの底面がBの上面±数mm かつ水平ズレ<1cm、gripper開放後1秒安定

### 1-4. 単体検証(ローカルPC)
- ランダムアクションで物理が破綻しないか
- `env.reset()` の初期配置が正しいか(可視化して目視)
- スクリプトで「掴んで持ち上げる」 scripted policy を書き、
  **学習なしでも物理的に積めること**を確認 → 環境バグを先に潰す(最重要)

---

## Phase 2: 学習(Google Colab Pro)

### 2-1. セットアップ
- `pip install mujoco gymnasium stable-baselines3[extra] sb3-contrib`
- コードはGitリポジトリ or Driveから同期。チェックポイントはDriveに定期保存

### 2-2. アルゴリズム
- 第一候補: **SAC + HER**(sb3-contribの`HerReplayBuffer`)。
  把持失敗エピソードも「別ゴールの成功例」に再利用でき、試行効率が大幅向上
- 代替: shaped報酬+PPO(実装が単純、収束は遅いが安定)

### 2-3. 並列化・スループット
- `SubprocVecEnv` で4〜8並列(ColabのCPUは2〜8vCPU、MuJoCo学習はCPUボトルネック)
- 目安: 1envあたり1〜3k step/s → 8並列で約10〜20k step/s
- 必要ステップ数の目安: HER+SACで **0.5M〜3M step**(数時間〜一晩)

### 2-4. 学習を安定させる仕掛け
- **カリキュラム**: 初期はA・Bを近接配置(数cm)→成功率が上がったら配置範囲を拡大
- **ドメインランダム化**(初日からON):
  - 摩擦 μ_slide ∈ [0.3, 1.0] / キューブ質量 ∈ [70, 105g] / 初期位置・姿勢ノイズ
  - アーム初期姿勢ノイズ / 観測ノイズ(数mm〜1cm) / アクション遅延1step混入
- コールバックで `eval成功率` を定期記録(TensorBoard → Colab上で確認)

### 2-5. 収束判定(Phase 2の出口)
- 評価100エピソードで **積み成功率 ≥ 80%** が3回連続
- ランダム化パラメータ全域(摩擦0.3〜1.0・質量・位置)で成功率 ≥ 70%
- 満たしたらポリシー重みを `.zip` で保存 → **ONNXエクスポート**(`torch.onnx`)

---

## Phase 3: Sim2Sim検証(ローカル / 任意だが推奨)

- 3-1. ローカルPCでONNXポリシーの推論テスト・動画保存(目視で動作の妥当性確認)
- 3-2. 既存のGazebo/ROS2環境(`dojo/lime2`)で同一ポリシーを駆動し、
       MuJoCo以外の物理エンジンでも積めるか確認 → 実機移植の事前リハーサル
- 3-3. 失敗したら: 失敗パターン(滑り・把持外れ・積み崩し)を分類し、
       ランダム化範囲/報酬を修正してPhase 2に戻る

---

## Phase 4: 実機デプロイ(現実世界)

### 4-1. 知覚: キューブの3D姿勢推定
- 既存資産: `yolo_ws` のキューブ検出+PnP姿勢推定、`barcode_detector` の実績
- 方針: RealSense D435i のRGB(+Depth)でキューブA/Bの位置姿勢を出し、
  アームベース座標系に変換してobsに供給
- 精度が厳しい場合のバックアップ: **キューブにArUcoマーカーを貼る**
  (既存のaruco_boxモデルと同じ手法で確実に姿勢が取れる)

### 4-2. ROS2推論ノード
- `onnxruntime` でポリシー実行。ループ10〜20Hz
- 入力: `/joint_states` + キューブ姿勢(4-1の出力) → simと同一のobs vectorを構成
- 出力: 関節位置目標 → `ros2_control` の position command interface
  (実機Dynamixelは位置制御なのでsim学習時の`<position>`アクチュエータと一致)
- ※ simと実機で **obs/行動の定義・単位・座標系を完全一致**させるのが成否の核心

### 4-3. 安全策(初回実行)
- 関節速度上限をsimの半分に制限、動作は低速から
- EEワークスペースをテーブル上の箱型領域に制限(外れたら即停止)
- 手を離せる位置で実行、異常時はトルクOFF

### 4-4. Sim2Realギャップの反復改善
- 実機ログ(joint_states・キューブ軌道)とsimログを比較:
  - キューブが滑る → frictionランダム化範囲を実測寄りに修正
  - 把持力不足・落下 → グリッパアクチュエータのgain/摩擦を調整
  - 位置ズレ系統誤差 → キャリブレーション or 観測ノイズ増やして再学習
- 1回の修正サイクル = 再学習(数時間)+ 実機テスト(数十回試行)

### 4-5. 最終成功基準
- キューブ2個をランダム配置 → **10試行中8回以上**、人手なしで2段積み完了
- 崩壊・落下時に安全停止できること

---

## マイルストーン一覧

| # | マイルストーン | 完了判定 |
|---|----------------|----------|
| M0 | 環境・モデル確認 | Phase 0(完了) |
| M1 | MuJoCo環境が動く | scripted policyで物理的に積める |
| M2 | 学習が収束 | sim成功率 ≥80% (ランダム化込み≥70%) |
| M3 | Sim2Sim検証 | Gazeboでも積める(任意) |
| M4 | 実機で単発成功 | 1回でも2段積み成功 |
| M5 | 実機で安定成功 | 10回中8回成功 |

## 主なリスクと対策

| リスク | 対策 |
|--------|------|
| 学習が収束しない | HER化 → 報酬分解デバッグ → カリキュラム幅を縮める |
| キューブ姿勢推定が実機で不安定 | ArUcoマーカー貼付にフォールバック |
| 実機でキューブが滑る/弾く | friction・restitutionのランダム化幅を実測で補正し再学習 |
| Colab無料枠/時間制限 | checkpoint定期保存で分割学習。Proなら24hセッション可 |
| mimic jointのMJCF変換ミス | URDF展開後にMuJoCoビューアで指の対称動作を目視確認 |
