from direct.showbase.ShowBase import ShowBase
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerQueue, CollisionRay
from panda3d.core import Material, LRotationf, NodePath
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import TextNode
from panda3d.core import LVector3, BitMask32
from direct.gui.OnscreenText import OnscreenText
from direct.interval.MetaInterval import Sequence, Parallel
from direct.interval.LerpInterval import LerpFunc
from direct.interval.FunctionInterval import Func, Wait
from direct.task.Task import Task
import sys

ACCEL = 70
MAX_SPEED = 5
MAX_SPEED_SQ = MAX_SPEED ** 2


class Game(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.tital = OnscreenText(text="Test", parent=base.a2dTopLeft, align=TextNode.ALeft, pos=(0.05, -0.08), fg=(1, 1, 1, 1), scale=0.06, shadow=(0, 0, 0, 0.5))
        self.accept("escape", sys.exit)


        self.disableMouse()
        camera.setPosHpr(0, 0, 25, 0, -90, 0)

        self.maze = loader.loadModel("models/maze")
        self.maze.reparentTo(render)


if __name__ == "__main__":
    game = Game()
    game.run()
