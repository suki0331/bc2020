from machine.car import drive
from machine.tv import watch

watch()
drive()

from machine import car
from machine import tv

car.drive()
tv.watch()

print ("=====================================")

from machine.test.car import drive
from machine.test.tv import watch

watch()
drive()

from machine.test import car
from machine.test import tv

car.drive()
tv.watch()

from machine import test
from machine import tv

test.car.drive()
test.tv.watch()