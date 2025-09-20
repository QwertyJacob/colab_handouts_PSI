# venn diagram in manim

from manim import *




class VennDiagram(Scene):
    def construct(self):

        ellipse1 = Ellipse(width=4, height=2, color=BLUE)
        ellipse2 = Ellipse(width=4, height=2, color=GREEN)

        # fill the ellipses
        ellipse1.set_fill(opacity=0.5)
        ellipse2.set_fill(opacity=0.5)

        ellipse1.move_to(LEFT * 3)
        ellipse2.move_to(RIGHT * 3)

        # lets put labels in the ellipses_
        label1 = MathTex("A").move_to(ellipse1.get_center())
        label2 = MathTex("B").move_to(ellipse2.get_center())

        self.play(Create(ellipse1), Create(ellipse2))
        self.play(Create(label1), Create(label2))
        self.wait(1)
        # lets move the right ellipse to the center of the page a bit:
        self.play(ellipse2.animate.move_to(ORIGIN + RIGHT * 1),
                  label2.animate.move_to(ORIGIN + RIGHT * 1),
                  label1.animate.move_to(ORIGIN + LEFT * 1))
        self.play(ellipse1.animate.move_to(
            ellipse2.get_center() - 2 * RIGHT))
        self.wait(12)

        
