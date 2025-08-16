class Object:
    def __init__(self, id: int, category: int, X_coordinate, Y_coordinate, width, height, conf: float, isTrue):
        self.__id = int(id)
        self.__category = int(category)
        try:  
            self.X_coordinate = float(X_coordinate)
            self.Y_coordinate = float(Y_coordinate)
            self.width = float(width)
            self.height = float(height)
            self.__conf = float(conf)
            if float(isTrue) >= 1:
                self.__isTrue = True
            elif float(isTrue) == 0:
                self.__isTrue = False
            else :
                print(isTrue, type(isTrue))
                raise SyntaxError('dataset error')
        except ValueError: #推論できなかった物体の処理
            self.X_coordinate = float('nan')
            self.Y_coordinate = float('nan')
            self.width = float('nan')
            self.height = float('nan')
            self.__conf = float(0)
            self.__isTrue = False
        
    @property
    def id(self):
        return self.__id
    @property
    def category(self):
        return self.__category
    @property
    def conf(self):
        return self.__conf
    @property
    def isTrue(self):
        return self.__isTrue
    @property
    def CameraID(self):
        return self.CameraID
    
    def CameraID(self, CameraID):
        self.CameraID = CameraID
    
    def info(self):
        print(self.__id,
              self.__category,
              self.X_coordinate,
              self.Y_coordinate,
              self.width,
              self.height,
              self.__conf,
              self.__isTrue
              )
