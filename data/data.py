class Employee():
    def __init__(self, a, b, c, d, e, f, g):
        self.rank : int = a # 직급
        self.task_score : dict = b #{'business' : 0, 'individual' : 0} # 경력지수
        self.distant_branch_list : list = c # 격지 지점 리스트, barnch index
        self.preferred_branch_list : list = d # 선호 지점 리스트, branch index
        self.worked_at_distant = e # 2년 내 격지 근무 여부
        self.worked_at_inferior = f # 2년 내 열악지점 근무 여부
        self.relative_list : list = g # 해당 지역본부에 배치될 직원 명단 중에 친척 리스트

class Branch():
    def __init__(self, a, b, c, d, e):
        self.required_personnel : int = a # TO
        self.required_vm : int = b # vm 인력 TO
        self.required_task_score : dict = c # 필요 역량 점수 딕셔너리
        self.required_rank : dict = d
        self.inferiorean = e #열악 지점 여부
        
        #현재 지점에 배치된 사람의 수 저장
        self.current_personnel : int = 0
        self.current_rank_num : dict = {4:  0, 5 : 0}
        self.current_task_score : dict = {'e': 0, 'i': 0}
        self.current_num_newbie : int = 0
            
    def reset(self):
        self.current_personnel : int = 0
        self.current_rank_num : dict = {4:  0, 5 : 0}
        self.current_task_score : dict = {'e': 0, 'i': 0}
        self.current_num_newbie : int = 0

e0 = Employee(5, {'e': 80, 'i': 100},[3],[0, 2],False,True,[6, 14]) 
e1 = Employee(5, {'e': 50, 'i': 60},[3],[1, 4],False,False,[]) 
e2 = Employee(4, {'e': 50, 'i': 60},[1,3],[1, 2],False,False, []) 
e3 = Employee(5, {'e': 90, 'i': 50},[2],[1, 4],False,False,[])
e4 = Employee(4, {'e': 90, 'i': 100},[4],[1, 0],True,False,[]) 
e5 = Employee(4, {'e': 60, 'i': 90},[3],[2, 4],False,True,[]) 
e6 = Employee(5, {'e': 40, 'i': 60},[0],[1, 3],False,False,[0, 14])
e7 = Employee(5, {'e': 60, 'i': 40},[4],[3, 2],False,False,[])
e8 = Employee(4, {'e': 70, 'i': 80},[3],[0, 4],True,True,[])
e9 = Employee(5, {'e': 60, 'i': 40},[3, 4],[2, 1],False,False,[]) 
e10 = Employee(5, {'e': 80, 'i': 50},[0],[3, 1],False,False,[11]) 
e11 = Employee(4, {'e': 50, 'i': 70},[0],[4, 1],True,False,[10])
e12 = Employee(5, {'e': 70, 'i': 80},[],[3, 1],False,False,[])
e13 = Employee(5, {'e': 70, 'i': 30},[0],[3, 2],False,False,[])
e14 = Employee(5, {'e': 100, 'i': 40},[4],[0, 3],False,False,[0, 6])
e15 = Employee(5, {'e': 150, 'i': 120},[2],[2, 1],False,False,[])
e16 = Employee(5, {'e': 90, 'i': 100},[1],[0, 1],False,False,[])
e17 = Employee(5, {'e': 120, 'i': 80},[3],[2, 1],False,False,[])

b0 = Branch(3, 1, {'e': 120,'i' : 150}, {4 : 1, 5 : 2}, 0)
b1 = Branch(2, 0, {'e': 100,'i' : 150}, {4 : 1, 5 : 1}, 0)
b2 = Branch(5, 2, {'e': 200,'i' : 250}, {4 : 1, 5 : 4}, 0)
b3 = Branch(3, 0, {'e': 180,'i' : 180}, {4 : 1, 5 : 2}, 1)
b4 = Branch(2, 0, {'e': 100,'i' : 100}, {4 : 1, 5 : 1}, 0)

employees = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
             e11, e12, e13, e14, e15, e16, e17]

branches = [b0, b1, b2, b3, b4]