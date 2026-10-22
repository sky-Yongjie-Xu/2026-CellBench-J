# sc_foundation_model_benchmark

统一启动：`python main.py --dataset pbmc12k`，默认依次使用全部 17 个已注册模型运行该数据集适用的全部任务。添加 `--cfgs configs/scgpt.yaml` 只运行该模型，添加 `--dry-run` 预览计划。任务日志与汇总在 `output/suites/`，用法详见[代码结构与统一入口](docs/code_structure.md)。

数据已统一为 `datasets/`，模型资源为 `weights/`。详见[资源路径与下载说明](docs/resources.md)。

运行环境：`conda activate cellbench`，详见[安装、验证与运行说明](docs/environment.md)。

评测直接本地运行，指标保存为 JSON；批量运行和 CSV 导出见[本地评测说明](docs/local_runs.md)。

新增六个通用模型的源码、权重、独立后端和验证状态见[新增模型接入说明](docs/frontier_models.md)。

全部模型的三阶段协议、与官方源码的差异和统一缓存接口见[适配器审查](docs/model_adapter_audit.md)。

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

评估指标已升级至 v2；定义、配置、结果迁移及验证边界见 [评估指标说明](docs/evaluation_metrics_v2.md)。

代码已按职责分层，目录、调用关系和迁移说明见 [代码结构](docs/code_structure.md)。
# 2026-CellBench-J
# 2026-CellBench-J
# 2026-CellBench-J
