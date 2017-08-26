#!/usr/bin/env python3.5

import pygame as pg
import sys
import pygame.locals as pg_local
import random
import time

class Board:

	def __init__(self, disp_surf = None, height = 480, width = 640, step = 20,
		 			back_color = pg.Color(186, 191, 198), 
	 				line_color = pg.Color(84, 81, 69), 
	 				line_thickness = 2):
		self._height = height
		self._width = width
		self._step = step
		self._back_color = back_color
		self._line_color = line_color
		self._disp_surf = disp_surf
		self._line_thickness = line_thickness

	def draw(self):
		self._disp_surf.fill(self._back_color)
		for i in range(0, self._width, self._step):
			pg.draw.line(self._disp_surf, self._line_color, (i, 0), (i, self._height), self._line_thickness)
		for i in range(0, self._height, self._step):
			pg.draw.line(self._disp_surf, self._line_color, (0, i), (self._width, i), self._line_thickness)

	def get_box_form(self):
		return  (int(self._width / self._step), int(self._height / self._step))

	def conv_left_top(self, boxx, boxy):
		return (boxx * self._step, boxy * self._step)

	def conv_box(self, left, top):
		return (int(left / self._step), int(top, self._step))

	def box_dim(self):
		return (self._step, self._step)

	def get_dim(self):
		return (self._width, self._height)


class Apple:
	def __init__(self, disp_surf, board):
		self._board = board
		self._disp_surf = disp_surf
		self._horizontal_boxes, self._vertical_boxes = self._board.get_box_form()
		self._coords_of_apple = self.__generate_random()
		self._color_inner = pg.Color(252, 92, 68)
		self._length,self._width = self._board.box_dim()
		self._color_outer = pg.Color(178, 68, 51)
		self.coords = self._coords_of_apple

	def __generate_random(self):
		random.seed(int(time.time()) % 1000)
		return (random.randint(5, self._horizontal_boxes-5), 
				random.randint(5, self._vertical_boxes-5))

	def draw(self, generate_new = False):
		if generate_new == True:
			self._coords_of_apple = self.coords = self.__generate_random()
		coords = self._board.conv_left_top(self._coords_of_apple[0], self._coords_of_apple[1])
		pg.draw.rect(self._disp_surf, self._color_inner, (coords[0]+2, coords[1]+2, self._length-2, self._width-2))
		pg.draw.rect(self._disp_surf, self._color_outer, (coords[0]+2, coords[1]+2, self._length-2, self._width-2), 4)


class Snake:
	def __init__(self, disp_surf, board):
		self._disp_surf = disp_surf
		self._board = board
		self._horizontal_boxes, self._vertical_boxes = self._board.get_box_form()
		self._length,self._width = self._board.box_dim()
		self.head = self.__generate_random()
		self.body = [{'x': self.head[0], 'y':self.head[1]},
					{'x': self.head[0]-1, 'y':self.head[1]},
					{'x': self.head[0]-2, 'y':self.head[1]}]
		self._color_inner = pg.Color(38, 168, 15)
		self._color_outer = pg.Color(36, 104, 17)
		self.cur_dir = 'right'

	def specify_head(self, coords):
		self.head = coords
		self.body = [{'x': self.head[0], 'y':self.head[1]},
					{'x': self.head[0]-1, 'y':self.head[1]},
					{'x': self.head[0]-2, 'y':self.head[1]}]
	def draw(self):
		for i in range(len(self.body)):
			coords = self._board.conv_left_top(self.body[i]['x'], self.body[i]['y'])
			pg.draw.rect(self._disp_surf, self._color_inner, (coords[0], coords[1], self._length, self._width))
			pg.draw.rect(self._disp_surf, self._color_outer, (coords[0], coords[1], self._length, self._width), 4)

	def __generate_random(self):
		random.seed(int(time.time()) % 1000)
		return (random.randint(10, self._horizontal_boxes-10), 
				random.randint(10, self._vertical_boxes-10))

	def cross_over(self):
		if self.head[0] < 0 or self.head[0] >= self._horizontal_boxes or \
		 self.head[1] < 0 or self.head[1] >= self._vertical_boxes:
		 	return True
		for i in range(1, len(self.body)):
			if self.body[i]['x'] == self.head[0] and self.body[i]['y'] == self.head[1]:
				return True
		return False

	def remove_tail(self):
		if len(self.body) != 0:
				self.body.pop()

	def push_coords(self, coords, on_tail = False):
		if on_tail == True:
			self.body.append({'x': coords[0], 'y':coords[1]})
			return
		self.body.insert(0, {'x': coords[0], 'y': coords[1]})
		self.head = coords

	def _get_possible(self, val):
		if val in ('up', 'down'):
			return ['left', 'right']
		elif val in ('left', 'right'):
			return ['up', 'down']

	def change_dir(self, new_dir):
		possibles = self._get_possible(self.cur_dir)
		if not new_dir in possibles:
			return
		else:
			self.cur_dir = new_dir
		self.make_changes()

	def make_changes(self, remove_tail_bool = True):
		if remove_tail_bool == True:
			self.remove_tail()
		to_add = None
		if self.cur_dir == 'up':
			to_add = (self.head[0], self.head[1] - 1)
		elif self.cur_dir == 'down': 
			to_add = (self.head[0], self.head[1] + 1)
		elif self.cur_dir == 'left':
			to_add = (self.head[0] - 1, self.head[1])
		elif self.cur_dir == 'right':
			to_add = (self.head[0] + 1, self.head[1])
		self.push_coords(to_add)

	def move(self, generate_tail = False):
		if generate_tail == False:
			self.make_changes()
		else:
			self.make_changes(remove_tail_bool = False)

class ComicWrites:
	def __init__(self, disp_surf, font_spec = "./fonts/font_to_use.ttf", font_size = 32):
		self.font_spec = font_spec
		self.font_size = font_size
		self.disp_surf = disp_surf
		self.__font_obj = pg.font.Font(self.font_spec, self.font_size)

	def get_write(self, to_write, coords, anti_alias = True, text_color = pg.Color(46, 173, 53)):
		text_surf_obj = self.__font_obj.render(to_write, anti_alias, text_color)
		text_rect_obj = text_surf_obj.get_rect()
		text_rect_obj.center = coords
		self.disp_surf.blit(text_surf_obj, text_rect_obj)
		return self.disp_surf

class SnakeGame:
	def __init__(self, frame_count = 30):
		pg.init()
		self.disp_surf = pg.display.set_mode((640, 480))
		pg.display.set_caption(("Twister"))
		self.board = Board(self.disp_surf)
		self.fps_clock = pg.time.Clock()
		self.FPS = frame_count
		self.apple = None
		self.snake = None
		self.score = 0

	def new_pieces(self):
		return (Apple(self.disp_surf, self.board), Snake(self.disp_surf, self.board), 0)

	def main_loop(self):
		self.start_game()
		while True:
			self.run()
			self.game_over()

	def run(self):
		self.apple, self.snake, self.score = self.new_pieces()
		while True:	
			for event in pg.event.get():
				if event.type == pg_local.QUIT:
					pg.quit()
					sys.exit()
				if event.type == pg_local.KEYUP:
					if event.key in (pg_local.K_LEFT, pg_local.K_a):
						self.snake.change_dir("left")
					elif event.key in (pg_local.K_RIGHT, pg_local.K_d):
						self.snake.change_dir("right")
					elif event.key in (pg_local.K_UP, pg_local.K_w):
						self.snake.change_dir('up')
					elif event.key in (pg_local.K_DOWN, pg_local.K_s):
						self.snake.change_dir('down')
			self.board.draw()
			self.snake.draw()
			if self.snake.cross_over():
				break
			if self.snake.head == self.apple.coords:
				self.apple.draw(generate_new = True)
				self.snake.move(generate_tail = True)
			else:
				self.apple.draw()
				self.snake.move()
			pg.display.update()
			self.fps_clock.tick(self.FPS)

	def game_over(self):
		dim = self.board.get_dim()
		game_over_surf = pg.Surface(dim)
		game_over_surf = game_over_surf.convert_alpha()
		back_color = pg.Color(0, 0, 0, 200)
		game_over_surf.fill(back_color)
		game_over_font = ComicWrites(game_over_surf, font_size = 60)
		game_over_surf = game_over_font.get_write("Game over", (dim[0]/2, dim[1]/2))
		self.disp_surf.blit(game_over_surf, (0, 0))
		while True:
			for event in pg.event.get():
				if event.type == pg_local.QUIT:
					pg.quit()
					sys.exit()
				if event.type == pg_local.KEYUP:
					return
			pg.display.update()

	def start_game(self):
		dim = self.board.get_dim()
		game_start_surf = pg.Surface(dim).convert_alpha()
		back_color = pg.Color(35, 147, 46, 200)
		game_start_surf.fill(back_color)
		game_start_font = ComicWrites(game_start_surf, font_size = 50)
		game_start_press_any_font = ComicWrites(game_start_surf, font_size = 20)
		game_start_surf = game_start_font.get_write("Snakky!", (dim[0]/2, dim[1]/2), text_color = pg.Color(125, 237, 127))
		game_start_surf = game_start_press_any_font.get_write("Press any key to continue...", (dim[0]/2, dim[1]-20))
		self.disp_surf.blit(game_start_surf, (0, 0))
		snake = Snake(game_start_surf, self.board)
		while True:
			for event in pg.event.get():
				if event.type == pg_local.QUIT:
					pg.quit()
					sys.exit()
				if event.type == pg_local.KEYUP:
					return
			pg.display.update()