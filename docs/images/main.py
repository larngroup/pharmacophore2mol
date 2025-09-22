###### TROQUEI AS CORES DIRETAMENTE NA INSTALAÇAO DO MANIM, NO VENV. P VOLTAR AO NORMAL VAI LA E COMENTA AS ALTERAÇOES
#usa iste import p ires la ter
from manim import WHITE


#############################################################33

from manim import *
from manim.opengl import *

config.frame_rate = 30  # optional, just an example
config.pixel_height = 1920 * 5
config.pixel_width = 1360 * 4


class PhenolVsFuran(Scene):
    def construct(self):
        hello_world = Text("Phenol vs Furan", font_size=72)
        self.play(Write(hello_world))
        # self.play(
        #     self.camera.animate.set_euler_angles(
        #         theta=-10*DEGREES,
        #         phi=30*DEGREES,
        #     )
        # )
        self.play(FadeOut(hello_world))



class PharmacophoreExample(Scene):
    def construct(self):
        img = ImageMobject("./docs/images/assets/meth_with_pharm.png").scale_to_fit_width(20)

        legend_container = VGroup()
    
        for i, (legend, color) in enumerate(zip(["Aromatic", "H-Bond Donnor", "H-Bond Acceptor"], [PURE_GREEN, PURE_RED, PURE_BLUE])):
            circle_center = Circle(radius=0.4, color=color, fill_opacity=1, stroke_width=2)
            text_center = Tex(legend + " Center", font_size=48)
            circle_dir = Circle(radius=0.2, color=color, fill_opacity=1, stroke_width=2)
            text_dir = Tex(legend + " Direction", font_size=48)
            text_center.next_to(circle_center, RIGHT, buff=0.5)
            circle_dir.next_to(circle_center, DOWN, buff=0.5)
            text_dir.next_to(circle_dir, RIGHT, buff=0.5).align_to(text_center, LEFT)
            inner_container = VGroup(circle_center, text_center, circle_dir, text_dir)
            if i > 0:
                inner_container.next_to(legend_container[i-1], DOWN, buff=1).align_to(legend_container[i-1], LEFT)
            legend_container.add(inner_container)

        legend_container.next_to(img, RIGHT, buff=1)

        main_container = Group(
            img,
            legend_container,
        )

        main_container.scale_to_fit_width(config.frame_width * 0.95)
        main_container.move_to(ORIGIN)
        self.add(main_container)


class PharmacophoreExampleRandom(Scene):
    def construct(self):
        img = ImageMobject("./docs/images/assets/random_pharm.png").scale_to_fit_width(20)

        legend_container = VGroup()
    
        for i, (legend, color) in enumerate(zip(["Aromatic", "H-Bond Donnor", "H-Bond Acceptor"], [PURE_GREEN, PURE_RED, PURE_BLUE])):
            circle_center = Circle(radius=0.4, color=color, fill_opacity=1, stroke_width=2)
            text_center = Tex(legend + " Center", font_size=48)
            circle_dir = Circle(radius=0.2, color=color, fill_opacity=1, stroke_width=2)
            text_dir = Tex(legend + " Direction", font_size=48)
            text_center.next_to(circle_center, RIGHT, buff=0.5)
            circle_dir.next_to(circle_center, DOWN, buff=0.5)
            text_dir.next_to(circle_dir, RIGHT, buff=0.5).align_to(text_center, LEFT)
            inner_container = VGroup(circle_center, text_center, circle_dir, text_dir)
            if i > 0:
                inner_container.next_to(legend_container[i-1], DOWN, buff=1).align_to(legend_container[i-1], LEFT)
            legend_container.add(inner_container)

        legend_container.next_to(img, RIGHT, buff=1)

        main_container = Group(
            img,
            legend_container,
        )

        main_container.scale_to_fit_width(config.frame_width * 0.95)
        main_container.move_to(ORIGIN)
        self.add(main_container)

class Unet2DArch(Scene):



    def high_level_arch(self):
        PAD = 0.5
        rect_w, rect_h = 2, 0.8

        mid_text = Tex("MidBlock", font_size=28)
        # mid_text = mid_text.scale_to_fit_width(rect_w * PAD)
        # if mid_text.height > rect_h * PAD:
        #     mid_text = mid_text.scale_to_fit_height(rect_h * PAD)
        mid_block = VGroup(mid_text, Rectangle(width=rect_w * 1.5, height=rect_h).set_stroke(WHITE, 2))
        downs = VGroup()
        for d in range(3):
            if d == 1:
                text = Tex(f" DownBlock\n\nw/ Attention", font_size=28)
            else:
                text = Tex(f"DownBlock", font_size=28)

            # text = text.scale_to_fit_height(rect_h * PAD)
            # if text.width > rect_w * PAD:
            #     text = text.scale_to_fit_width(rect_w * PAD)
            block = VGroup(text, Rectangle(width=rect_w, height=rect_h).set_stroke(WHITE, 2))
            if d != 0:
                block.next_to(downs[-1], DOWN, buff=0.5)
                block.next_to(block, RIGHT, buff=-1)
                

            downs.add(block)
        downs.to_corner(UP + LEFT)

        ups = VGroup()

        for u in range(3):
            if u == 1:
                text = Tex(f"    UpBlock\n\nw/ Attention", font_size=28)
            else:
                text = Tex(f"   UpBlock", font_size=28)

            block = VGroup(text, Rectangle(width=rect_w, height=rect_h).set_stroke(WHITE, 2))
            if u != 0:
                block.next_to(ups[-1], DOWN, buff=0.5)
                block.next_to(block, LEFT, buff=-1)
            ups.add(block)

        target_pos = mid_block.get_corner(UP + RIGHT)
        ups.shift(target_pos - ups[2].get_bottom() - 0.5 * DOWN + 0.5 * RIGHT)
        target_pos = mid_block.get_corner(UP + LEFT)
        downs.shift(target_pos - downs[2].get_bottom() - 0.5 * DOWN + 0.5 * LEFT)


        #input/output
        opar = Tex("(", font_size=256)
        embt = Tex(", emb(t))", font_size=48)
        cpar = Tex(")", font_size=256)
        input_label = Tex(
            """
            (Noisy voxel grid grid at timestep t)
            """,
            font_size=28)
        input_image = ImageMobject("./docs/images/assets/noisy_phenol.png").scale_to_fit_height(3.2)
        
        opar.next_to(input_image, LEFT, buff=0.5)
        embt.next_to(input_image, RIGHT, buff=0.5)
        cpar.next_to(embt, RIGHT, buff=0)
        input_container = Group(input_image, opar, embt, cpar)
        input_container.next_to(downs[0], LEFT, buff=2)
        input_striped = input_container.copy()
        input_container.add(input_label.next_to(input_image, DOWN, buff=0.5))
        in_container_label = Tex("Shape:\n\n((\#channels, L, L[, L]), dim(emb(t)))", font_size=32).next_to(input_container, DOWN, buff=1.5)
        in_container_sublabel = Tex("where L is the number of voxels used", font_size=24).next_to(in_container_label, DOWN, buff=0.5)
        input_container.add(in_container_label, in_container_sublabel)


        output_label = Tex("(Total noise added until t)", font_size=28)
        output_image = ImageMobject("./docs/images/assets/noise_pred.png").scale_to_fit_height(3.2)
        output_image.next_to(ups[0], RIGHT, buff=5)
        output_container = Group(output_label.next_to(output_image, DOWN, buff=0.5), output_image)
        out_container_label = Tex("Shape:\n\n(\#channels, L, L[, L])", font_size=32)
        out_container_label.next_to(output_container, DOWN, buff=1.5)
        output_container.add(out_container_label)

        arch_blocks = Group(
            mid_block,
            downs,
            ups,
            input_container,
            output_container,
        )

        

        #arrows
        arrows = VGroup()
        for b in range(len(downs)):
            if b != 0:
                mid_pt = midpoint(downs[b-1].get_bottom(), downs[b].get_top())
                end_pos = (mid_pt[0], downs[b].get_top()[1], 0)
                start_pos = (mid_pt[0], downs[b-1].get_bottom()[1], 0)
                arrow = Arrow(start_pos, end_pos, buff=0, max_tip_length_to_length_ratio=0.5)
                arrows.add(arrow)
        for b in range(len(ups)):
            if b != 0:
                mid_pt = midpoint(ups[b-1].get_top(), ups[b].get_bottom())
                end_pos = (mid_pt[0], ups[b-1].get_bottom()[1], 0)
                start_pos = (mid_pt[0], ups[b].get_top()[1], 0)
                arrow = Arrow(start_pos, end_pos, buff=0, max_tip_length_to_length_ratio=0.5)
                arrows.add(arrow)

        arrows.add(
            Arrow(
                start=(
                    mid(downs[-1].get_corner(DOWN + RIGHT), mid_block.get_corner(UP + LEFT))[0],
                    downs[-1].get_bottom()[1],
                    0
                ),
                end=(
                    mid(downs[-1].get_corner(DOWN + RIGHT), mid_block.get_corner(UP + LEFT))[0],
                    mid_block.get_top()[1],
                    0
                ),
                buff=0,
                max_tip_length_to_length_ratio=0.5
            ),
            Arrow(
                start=(
                    mid(ups[-1].get_corner(DOWN + LEFT), mid_block.get_corner(UP + RIGHT))[0],
                    mid_block.get_top()[1],
                    0
                ),
                end=(
                    mid(ups[-1].get_corner(DOWN + LEFT), mid_block.get_corner(UP + RIGHT))[0],
                    ups[-1].get_bottom()[1],
                    0
                ),
                buff=0,
                max_tip_length_to_length_ratio=0.5
            )
        )
        skips = VGroup()
        for s in range(len(downs)):
            skip = LabeledArrow(stroke_width=1, tip_shape=None, label="concat", label_config={"font_size": 24}, start=downs[s].get_right(), end=ups[s].get_left(), frame_config={"stroke_width": 0})
            skips.add(skip)

        in_arrow = LabeledArrow(label="Conv", label_config={"font_size": 24}, frame_config={"stroke_width": 0}, stroke_width=1, start=input_striped.get_right(), end=downs[0].get_left(), buff=0.1, max_stroke_width_to_length_ratio=100, max_tip_length_to_length_ratio=0.5)
        arrows.add(in_arrow)
        out_arrow = LabeledArrow(label="GroupNorm(SiLU(Conv))", label_config={"font_size": 24}, frame_config={"stroke_width": 0}, stroke_width=1, start=ups[0].get_right(), end=output_image.get_left(), buff=0.1, max_stroke_width_to_length_ratio=100, max_tip_length_to_length_ratio=0.5)
        arrows.add(out_arrow)
        arch = Group(
            arch_blocks,
            arrows,
            skips,
        )
        title = Text("Base Model Architecture", font_size=48)
        arch_container = Group(title, arch.next_to(title, DOWN, buff=2.5))
        return arch_container
    

    def conditioning_encoder(self):
        PAD = 0.5
        rect_w, rect_h = 2, 0.8


        downs = VGroup()
        for d in range(4):
            if d == 1 or d == 3:
                text = Tex(f" DownBlock\n\nw/ Attention", font_size=28)
            else:
                text = Tex(f"DownBlock", font_size=28)

            block = VGroup(text, Rectangle(width=rect_w, height=rect_h).set_stroke(WHITE, 2))
            if d != 0:
                block.next_to(downs[-1], DOWN, buff=0.5)
                block.next_to(block, RIGHT, buff=-1)
                

            downs.add(block)
        downs.to_corner(UP + LEFT)



        #input/output
        input_label = Tex(
            """
            (Pharmacophore conditioning)
            """,
            font_size=28)
        input_image = ImageMobject("./docs/images/assets/phenol_pharm_slice.png").scale_to_fit_height(3.2)
        
        input_container = Group(input_image)
        input_container.next_to(downs[0], LEFT, buff=2)
        input_striped = input_container.copy()
        input_container.add(input_label.next_to(input_image, DOWN, buff=0.5))
        in_container_label = Tex("Shape:\n\n(\#channels, L, L[, L]),", font_size=32).next_to(input_container, DOWN, buff=1.5)
        in_container_sublabel = Tex("where L is the number of voxels used", font_size=24).next_to(in_container_label, DOWN, buff=0.5)
        input_container.add(
            in_container_label,
            in_container_sublabel,
            )


        arch_blocks = Group(
            downs,
            input_container,
        )

        

        #arrows
        arrows = VGroup()
        r = 0
        for b in range(len(downs)):
            if b != 0:
                mid_pt = midpoint(downs[b-1].get_bottom(), downs[b].get_top())
                end_pos = (mid_pt[0], downs[b].get_top()[1], 0)
                start_pos = (mid_pt[0], downs[b-1].get_bottom()[1], 0)
                arrow = Arrow(start_pos, end_pos, buff=0, max_tip_length_to_length_ratio=0.5)
                arrows.add(arrow)
            
            
            if b == 1 or b == 3:
                r += 1
                out_label = Tex(f"to cross attention at resolution $r_{r}$", font_size=24)
                out_label.next_to(downs[b], RIGHT, buff=2)
                arch_blocks.add(out_label)
                arrows.add(
                    Arrow(
                        start=downs[b].get_right(),
                        end=out_label.get_left(),
                        stroke_width=1,
                    )
                )


        in_arrow = LabeledArrow(label="Conv", label_config={"font_size": 24}, frame_config={"stroke_width": 0}, stroke_width=1, start=input_striped.get_right(), end=downs[0].get_left(), buff=0.1, max_stroke_width_to_length_ratio=100, max_tip_length_to_length_ratio=0.5)
        arrows.add(in_arrow)

        cond = Group(
            arch_blocks,
            arrows,
        )
        title = Text("Conditioning Encoder Architecture", font_size=48)
        cond_container = Group(title, cond.next_to(title, DOWN, buff=2.5))
        return cond_container

    def block_arch(self):

        rect_w, rect_h = 2, 0.8
        block_container = Group()
        title = Text("Down/Up/Middle Blocks", font_size=48)
        
        text = Tex("ResNetv2\n\nBlock", font_size=28)
        rect = Rectangle(height=rect_h, width=rect_w, stroke_width=1)
        resnet_container1 = VGroup(text, rect)

        opar = Tex("[", font_size=256)
        cpar = Tex("]", font_size=256)
        s_attn_text = Tex("Self\n\nAttention", font_size=28)
        s_attn_rect = Rectangle(height=rect_w, width=rect_h + 0.5, stroke_width=1)
        s_attn_container = VGroup(s_attn_text, s_attn_rect)
        c_attn_text = Tex("Cross\n\nAttention", font_size=28)
        c_attn_rect = Rectangle(height=rect_w, width=rect_h + 0.5, stroke_width=1)
        c_attn_container = VGroup(c_attn_text, c_attn_rect)
        s_attn_container.next_to(opar, RIGHT, buff=0)
        cpar.next_to(s_attn_container, RIGHT, buff=0)
        opar2 = opar.copy()
        cpar2 = cpar.copy()
        opar2.next_to(cpar, RIGHT, buff=0.1)
        c_attn_container.next_to(opar2, RIGHT, buff=0)
        cpar2.next_to(c_attn_container, RIGHT, buff=0)
        attn_container = VGroup(
            s_attn_container, opar, cpar, c_attn_container, opar2, cpar2
        )
        attn_container.next_to(resnet_container1, RIGHT, buff=0.1)

        resnet_container2 = resnet_container1.copy()
        resnet_container2.next_to(cpar2, RIGHT, buff=0.1)

        resize_text = Tex("AvgPooling\n\n(DownBlocks) \n\nOR\n\nConvTranspose\n\n(UpBlocks)", font_size=28)
        resize_rect = Rectangle(height=rect_w + 1, width=rect_h + 1.5, stroke_width=1)
        resize_container = VGroup(resize_text, resize_rect)
        resize_container.next_to(resnet_container2, RIGHT, buff=0.1)

        # legend = Tex("Self and cross attention presence is dependent on which experiment and layer is considered. When present, they are applied in this order.", font_size=24)
        
        block_container.add(
            resize_container,
            resnet_container1,
            resnet_container2,
            attn_container,
        )
        block_container.add(SurroundingRectangle(block_container, buff=0.5, color=WHITE, stroke_opacity=0.2, stroke_width=4))
        in_arrow = Arrow(start=block_container.get_left() + LEFT * 1.5, end=resnet_container1.get_left(), buff=0.2, stroke_width=1)
        out_arrow = Arrow(start=resize_container.get_right(), end=block_container.get_right() + RIGHT * 1.5, buff=0.2, stroke_width=1)
        attn_input = Tex("$enc(conditioning)_r$", font_size=24)
        attn_input.next_to(c_attn_container, UP, buff=2)
        attn_input_arrow = Arrow(start=attn_input.get_bottom(), end=c_attn_container.get_top(), buff=0.25, stroke_width=1)
        attn_input_container = VGroup(attn_input)
        block_container.add(attn_input_container, attn_input_arrow)
        block_container.add(in_arrow, out_arrow)
        # legend.next_to(block_container, DOWN, buff=0.5)
        title.next_to(block_container, UP, buff=1)
        block_container.add(
            title,
            # legend,
        )
        return block_container
    
    def resnetv2(self):
        resnet_container = VGroup()
        h = Tex("h", font_size=28)
        norm1 = Tex("GroupNorm", font_size=28)
        act1 = Tex("SiLU", font_size=28)
        conv1 = Tex("Conv", font_size=28)
        sum1 = Tex("$\oplus$", font_size=64)
        norm2 = Tex("GroupNorm", font_size=28)
        act2 = Tex("SiLU", font_size=28)
        conv2 = Tex("Conv", font_size=28)
        sum2 = Tex("$\oplus$", font_size=64)
        resnet_container.add(
            h,
            norm1,
            act1,
            conv1,
            sum1,
            norm2,
            act2,
            conv2,
            sum2,
        ).arrange(RIGHT, buff=1.5)
        for i in range(len(resnet_container)):
            if i != 0:
                start = resnet_container[i-1].get_right()
                end = resnet_container[i].get_left()
                arrow = Arrow(start, end, buff=0.25, stroke_width=1,)
                resnet_container.add(arrow)
        shortcut = ArcBetweenPoints(start=h.get_bottom() + 0.1 * DOWN + 0.17 * RIGHT, end=sum2.get_bottom() + 0.05 * DOWN + 0.25 * LEFT, stroke_width=1, radius=25)
        shortcut.add_tip(tip_shape=ArrowTriangleFilledTip, tip_width=0.25, tip_length=0.3)
        resnet_container.add(shortcut)
        emb_t = Tex("emb(t)", font_size=28)
        emb_t.next_to(resnet_container, UP, buff=1.5)
        emb_t.move_to((sum1.get_x(), emb_t.get_y(), 0))
        emb_arrow = Arrow(start=emb_t.get_bottom(), end=sum1.get_top(), buff=0.25, stroke_width=1)
        rect = SurroundingRectangle(resnet_container, buff=0.8, color=WHITE, stroke_opacity=0.2, stroke_width=4)
        resnet_container.add(rect)
        resnet_container.add(emb_t)
        resnet_container.add(emb_arrow)
        resnet_container.add(
            Arrow(start=((resnet_container.get_left() + LEFT * 1.5)[0], h.get_y(), 0), end=h.get_left(), buff=0.25, stroke_width=1),
            Arrow(start=sum2.get_right(), end=((resnet_container.get_right() + RIGHT * 1.5)[0], sum2.get_y(), 0), buff=0.25, stroke_width=1)
        )

        title = Text("ResNetv2 Block", font_size=48)
        title.next_to(resnet_container, UP, buff=1)
        resnet_container.add(title)

        return resnet_container

    def construct(self):


        main_container = Group()

        arch_container = self.high_level_arch()

        conditioning_encoder = self.conditioning_encoder()
        conditioning_encoder.next_to(arch_container, DOWN, buff=3)

        block_container = self.block_arch()
        block_container.next_to(conditioning_encoder, DOWN, buff=3)

        resnet_container = self.resnetv2()
        resnet_container.next_to(block_container, DOWN, buff=3)

        main_container.add(
            arch_container, 
            conditioning_encoder,
            block_container,
            resnet_container,
        )
        main_container.scale_to_fit_width(config["frame_width"] * 0.95)
        # if main_container.height > (config["frame_height"] * 0.95):
        #     main_container.scale_to_fit_height(config["frame_height"] * 0.95)
        main_container.move_to(ORIGIN)
        self.add(main_container)#, SurroundingRectangle(main_container))