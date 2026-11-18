# TurtleBot3 Lime: 実機関連資料（LimeSimulDemoとの比較）

確認日: 2026-11-18

## 重要な区別
- **LimeSimulDemo**: Gazebo上でTurtleBot3 Limeの探索・把持を試す公開デモ。PC単体で動作する。
  https://github.com/momoiorg-repository/LimeSimulDemo
- **LimeDemo（実機用）**: 開発者サイトで言及されるが、2026-11-18に公開リンクを確認した際、 https://github.com/momoiorg-repository/LimeDemo は404だった。公開されていない・移動した・権限が必要など可能性があり、存在しないと断定はできない。
  https://momoi.org/?yada_wiki=limedemo
- **ROBOTIS公式Limeパッケージ**: ROS 2 Humble対応Lime用のパッケージおよびハードウェア接続実装の調査起点。
  https://github.com/ROBOTIS-JAPAN-GIT/turtlebot3_lime

## 実機ハードウェアの概要
開発元の説明によるLimeの特徴:
- TurtleBot3 Waffleをベースにした移動マニピュレータ
- Jetson Orin Nano
- 6軸アーム
- RealSense D435i深度カメラ
- 車輪径の拡大
- ROS 2 Humble

出典: https://momoi.org/?yada_wiki=lime
製品: https://e-shop.robotis.co.jp/product.php?id=531

## デモで想定される処理
LimeSimulDemoには Behavior Tree (bt_search / bt_catch / bt_pick) があり、Gazebo内のコーラ缶の探索、接近、把持を行う。Nav2、MoveIt 2、カメラ解析、ros_actorを利用。
- https://github.com/momoiorg-repository/LimeSimulDemo
- https://momoi.org/?yada_wiki=limedemo

**注意**: Gazebo上のコーラ缶質量を0.39 kgから0.039 kgに変更するのはシミュレーションモデルの操作であり、実機の安全な可搬質量の証拠ではない。

## 実機移植のための確認チェック
1. 実機と開発PCのネットワーク、ROS_DOMAIN_ID、ROS 2 RMW実装が一致しているか
2. ロボットのUSB/シリアルポートおよびDYNAMIXELの電源・接続状態
3. ros2_controlのハードウェアプラグイン、jointとstate/command interfaceの定義一致
4. 実機のアーム・車輪・カメラの各ROSトピックが利用可能か
5. Gazebo固有の起動ノード、センサー、world依存処理が実機起動と混在していないか
6. 初回動作は障害物のない場所で低速かつ非常停止可能な状態で検証する

## 既出ログに関する注意
`Not connected manipulator` → `Failed to initialize hardware 'TurtleBot3LimeSystem'` → `missing state/command interfaces` の順で報告されている場合、まずマニピュレータ接続/初期化の失敗を調べる。interface欠落は初期化失敗の二次的症状である可能性がある。ログだけで接続ケーブル、電源、デバイスパスなどの原因は確定できない。

## 追加参考
- ROBOTIS公式 TurtleBot3 ROSパッケージ: https://github.com/ROBOTIS-GIT/turtlebot3
- ROBOTIS TurtleBot3 Manipulationマニュアル（一般的なTurtleBot3 + OpenMANIPULATOR向け。Lime固有手順と同一とは限らない）:
  https://docs.robotis.com/docs/systems/turtlebot3/manipulation/
- 開発元の紹介: https://momoi.org/?yada_wiki=lime

## 未確認項目
- LimeDemo実機用の現在公開されている完全なソース
- ユーザー所有の実機のファームウェア/ポート/シリアル構成
- このリポジトリの各launch設定との行単位の差分
- 実機実行での動作確認
