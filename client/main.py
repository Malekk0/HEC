import imghdr
from kivy.network.urlrequest import UrlRequest
from kivy.properties import BooleanProperty
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.widget import Widget
from kivymd.app import MDApp
from kivy.graphics import Color, Line
from requests_toolbelt import MultipartEncoder
from kivymd.uix.behaviors.toggle_behavior import MDToggleButton
from kivymd.uix.button import MDRaisedButton


class MyWidget(AnchorLayout):
    pass


class MyToggleButton(MDRaisedButton, MDToggleButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_down = self.theme_cls.primary_light


class UseRubber(Widget):
    rubber_active = BooleanProperty(True)

    def on_touch_down(self, touch):
        if not self.rubber_active:
            return
        ds = self.drawing_space
        if ds.collide_point(touch.x, touch.y):
            color = (0, 0, 1)
            with ds.canvas:
                Color(*color, mode='hsv')
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=15)

    def on_touch_move(self, touch):
        ds = self.drawing_space
        if 'line' in touch.ud and ds.collide_point(touch.x, touch.y):
            touch.ud["line"].points += [touch.x, touch.y]


class DrawingSpace(Widget):
    pass


class DrawPencil(Widget):
    pencil_active = BooleanProperty(True)

    def on_touch_down(self, touch):
        if not self.pencil_active:
            return
        ds = self.drawing_space
        if ds.collide_point(touch.x, touch.y):
            color = (0, 0, 0)
            with ds.canvas:
                Color(*color, mode='hsv')
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=4)

    def on_touch_move(self, touch):
        ds = self.drawing_space
        if 'line' in touch.ud and ds.collide_point(touch.x, touch.y):
            touch.ud["line"].points += [touch.x, touch.y]


class MyApp(MDApp):
    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        self.url = 'https://handwrittenexpression.herokuapp.com/image'
        self.data_dir = getattr(self, 'user_data_dir')

    def build(self):
        return MyWidget()

    def clear(self, *args):
        self.root.ids._drawing_space.canvas.clear()
        self.root.ids.textl.text = ''

    def save(self, *args):
        self.root.ids._drawing_space.export_to_png(self.data_dir + 'image.png')

    def send_file(self, result):
        with open(self.data_dir + 'image.png', 'rb') as f:
            my_image = f.read()
            image_type = imghdr.what(None, h=my_image)
        payload = MultipartEncoder(
            fields={
                'file': (
                    'image.png',
                    my_image,
                    image_type
                )
            }
        )
        headers = {
            'Content-Type': payload.content_type
        }

        self.req = UrlRequest(
            self.url,
            on_success=self.res,
            req_body=payload,
            req_headers=headers,
            on_error=self.error
        )

    def res(self, *args):
        d = self.req.result
        self.root.ids.textl.text = ('Распознано: ' + d['Распознано'] +
                                    '\n' + 'Вычислено: ' + d['Вычислено'])

    def error(self, *args):
        self.root.ids.textl.text = 'Соединение не установлено'


my = MyApp()
my.run()
