# Installation

1. Follow the steps as normal
2. Uninstall ai2thor
3. Install "old" ai2thor from our fork
   4. `git clone https://github.com/Velythyl/ai2thor.git -b multiagentproc`
   5. `cd ai2thor`
   6. `pip3 install -e .`
   7. Why do it this way instead of `pip git+https://github.com/Velythyl/ai2thor.git@multiagentproc`? Because the repo is HUGE (>2GB) so its safer to do it "transparently" though git vs through pip.
