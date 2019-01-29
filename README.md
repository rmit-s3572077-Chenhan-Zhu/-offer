<h3>1.二维数组中的查找：在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。</h3>

两种思路
一种是：
把每一行看成有序递增的数组，
利用二分查找，
通过遍历每一行得到答案，
时间复杂度是nlogn
```public class Solution {
    public boolean Find(int [][] array,int target) {
         
        for(int i=0;i<array.length;i++){
            int low=0;
            int high=array[i].length-1;
            while(low<=high){
                int mid=(low+high)/2;
                if(target>array[i][mid])
                    low=mid+1;
                else if(target<array[i][mid])
                    high=mid-1;
                else
                    return true;
            }
        }
        return false;
 
    }
}
 ```
 
另外一种思路是：
利用二维数组由上到下，由左到右递增的规律，
那么选取右上角或者左下角的元素a[row][col]与target进行比较，
当target小于元素a[row][col]时，那么target必定在元素a所在行的左边,
即col--；
当target大于元素a[row][col]时，那么target必定在元素a所在列的下边,
即row++；

```
public class Solution {
    public boolean Find(int [][] array,int target) {
        int row=0;
        int col=array[0].length-1;
        while(row<=array.length-1&&col>=0){
            if(target==array[row][col])
                return true;
            else if(target>array[row][col])
                row++;
            else
                col--;
        }
        return false;
 
    }
}
```






<h3>2.替换空格：
请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。</h3>

/*
问题1：替换字符串，是在原来的字符串上做替换，还是新开辟一个字符串做替换！
问题2：在当前字符串替换，怎么替换才更有效率（不考虑java里现有的replace方法）。
      从前往后替换，后面的字符要不断往后移动，要多次移动，所以效率低下
      从后往前，先计算需要多少空间，然后从后往前移动，则每个字符只为移动一次，这样效率更高一点。
*/
解法1：



```public class Solution {
    public String replaceSpace(StringBuffer str) {
        int spacenum = 0;//spacenum为计算空格数
        for(int i=0;i<str.length();i++){
            if(str.charAt(i)==' ')
                spacenum++;
        }
        int indexold = str.length()-1; //indexold为为替换前的str下标
        int newlength = str.length() + spacenum*2;//计算空格转换成%20之后的str长度
        int indexnew = newlength-1;//indexold为为把空格替换为%20后的str下标
        str.setLength(newlength);//使str的长度扩大到转换成%20之后的长度,防止下标越界
        for(;indexold>=0 && indexold<newlength;--indexold){ 
                if(str.charAt(indexold) == ' '){  //
                str.setCharAt(indexnew--, '0');
                str.setCharAt(indexnew--, '2');
                str.setCharAt(indexnew--, '%');
                }else{
                    str.setCharAt(indexnew--, str.charAt(indexold));
                }
        }
        return str.toString();
    }
}
```

```
解法二:
public String replaceSpace(StringBuffer str) {
    	if(str==null) return null;
    	StringBuffer s = new StringBuffer();
    	for(int i = 0;i<str.length();i++) {
    		if(str.charAt(i)==' ') {
    			s.append("%20");
    				
    		}
    		else s.append(str.charAt(i));
    	}
    	return s.toString();
		
    }
```


<h3>3.从头到尾打印链表：输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。</h3>
解法一

















```
public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
		Stack<Integer> stack = new Stack<Integer>(); //借助于栈
		while(listNode!=null) {
			stack.push(listNode.val);
			listNode=listNode.next;
		}
		 ArrayList<Integer> res = new  ArrayList<Integer>();
		 while(!stack.isEmpty()) {
			 int v=stack.pop();
			 res.add(v);
		 }
		 return res;
        
    }

```


<h3>4.重建二叉树 ：输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。</h3>





















```
public class Solution {
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
        TreeNode root=reConstructBinaryTree(pre,0,pre.length-1,in,0,in.length-1);
        return root;
    }
    //前序遍历{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}
    private TreeNode reConstructBinaryTree(int [] pre,int startPre,int endPre,int [] in,int startIn,int endIn) {
         
        if(startPre>endPre||startIn>endIn)
            return null;
        TreeNode root=new TreeNode(pre[startPre]);
         
        for(int i=startIn;i<=endIn;i++)
            if(in[i]==pre[startPre]){
                root.left=reConstructBinaryTree(pre,startPre+1,startPre+i-startIn,in,startIn,i-1);
                root.right=reConstructBinaryTree(pre,i-startIn+startPre+1,endPre,in,i+1,endIn);
                      break;
            }
                 
        return root;
    }
}
```














<h3>5. 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。</h3>

```
import java.util.Stack;

public class Solution {
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
    
    public void push(int node) {
        stack1.push(node);
    }
    
    public int pop() {
            if(stack2.isEmpty()){
                while(!stack1.isEmpty()){
                stack2.push(stack1.pop());
        }
    }
    return stack2.pop();
}}

```













<h3>6. 旋转数组的最小数字：把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。</h3>





```import java.util.ArrayList;
import java.util.Arrays;
public class Solution {
    public int minNumberInRotateArray(int [] array) {
				if(array==null ) return 0;
				int low =0; int high=array.length-1;
				while(low<high) {
					int mid =(high+low)/2;;
					if(array[mid]>array[high]) {
						low = mid+1;
					}
					else if(array[mid]<array[high]) {
						high=mid;
					}
					else high=high-1;
				}
				return array[low];
    }
	
}



```















































<h3>7. 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
n<=39。</h3>

```
public class Solution {
    public int Fibonacci(int n) {
        int preNum=1;
        int prePreNum=0;
        int result=0;
        if(n==0)
            return 0;
        if(n==1)
            return 1;
        for(int i=2;i<=n;i++){
            result=preNum+prePreNum;
            prePreNum=preNum;
            preNum=result;
        }
        return result;
 
    }
}
```
































<h3>8. 矩形覆盖：我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？</h3>

依旧是斐波那契数列

2*n的大矩形，和n个2*1的小矩形

其中target*2为大矩阵的大小

target <= 0 大矩形为<= 2*0,直接return 1；

target = 1大矩形为2*1，只有一种摆放方法，return1；

target = 2 大矩形为2*2，有两种摆放方法，return2；

target = n 分为两步考虑：

      1.  第一次摆放一块 2*1 的小矩阵，则摆放方法总共为f(target - 1)
      
2.第一次摆放一块1*2的小矩阵，则摆放方法总共为f(target-2)

 因为，摆放了一块1*2的小矩阵（用√√表示），对应下方的1*2（用××表示）摆放方法就确定了，所以为f(targte-2)


```
public class Solution {
    public int RectCover(int target) {
      if(target  <= 1){
            return 1;
        }
        if(target*2 == 2){
            return 1;
        }else if(target*2 == 4){
            return 2;
        }else{
            return RectCover((target-1))+RectCover(target-2);
        }
    }
}
```


















<h3>9.二进制中1的个数：输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
</h3>

```
 //---------------正解--------------------------------
    //思想：用1（1自身左移运算，其实后来就不是1了）和n的每位进行位与，来判断1的个数
    private static int NumberOf1_low(int n) {
        int count = 0;
        int flag = 1;
        while (flag != 0) {
            if ((n & flag) != 0) {
                count++;
            }
            flag = flag << 1;
        }
        return count;
    }
    //--------------------最优解----------------------------
    public static int NumberOf1(int n) {
        int count = 0;
        while (n != 0) {
            ++count;
            n = (n - 1) & n;
        }
        return count;
    }

如果一个整数不为0，那么这个整数至少有一位是1。如果我们把这个整数减1，那么原来处在整数最右边的1就会变为0
，原来在1后面的所有的0都会变成1(如果最右边的1后面还有0的话)。其余所有位将不会受到影响。
举个例子：一个二进制数1100，从右边数起第三位是处于最右边的一个1。减去1后，第三位变成0，
它后面的两位0变成了1，而前面的1保持不变，因此得到的结果是1011.我们发现减1的结果是把最
右边的一个1开始的所有位都取反了。这个时候如果我们再把原来的整数和减去1之后的结果做与
运算，从原来整数最右边一个1那一位开始所有位都会变成0。如1100&1011=1000.也就是说，把
一个整数减去1，再和原整数做与运算，会把该整数最右边一个1变成0.那么一个整数的二进制
有多少个1，就可以进行多少次这样的操作。
```    












<h3>10.数值的整数次方：给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。</h3>

第一种
```
public double Power(double base, int exponent) {
      
        double res =1;
        for(int i =0;i<Math.abs(exponent);i++){
            res*=base;
        }
        if(exponent<0) {
        	return 1/res;
        }
        return res;
	  }

```










第二种

```
public:
    double Power(double base, int exponent) {
        long long p = abs((long long)exponent);
      double r = 1.0;
        while(p){
            if(p & 1) r *= base;
            base *= base;
            p >>= 1;
        }
        return exponent < 0 ? 1/ r : r;
    }
```
















<h3>11.调整数组顺序奇数在前偶数在后：输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。</h3>

```
public static void reOrderArray(int [] array) {
        if(array==null) return;
        int temp=0;
        for(int i = 1;i<array.length;i++) {
        	if(array[i]%2==1)  {
        	for(int j = i;j>0 ;j--) {
        		if(array[j-1]%2==0) {
        		int t =array[j];
        		array[j]=array[j-1];
        		array[j-1]=t;
        		}
        	}
        }}
        
    }
    
    
冒泡排序：

public class Solution {
    public void reOrderArray(int [] array) {
        for(int i=0;i<array.length-1;i++)
            for(int j=0;j<array.length-i-1;j++){
                if(array[j]%2==0 && array[j+1]%2==1){
                    int temp=array[j];
                    array[j]=array[j+1];
                    array[j+1]=temp;
                }
            }
    }
}
```


<h3>12.链表的倒数k个结点：输入一个链表，输出该链表中倒数第k个结点。</h3>

















```

public ListNode FindKthToTail(ListNode head,int k) {
        ListNode fast=head;
        ListNode slow = head;
        int count=0;
        
        while(fast!=null&&count!=k){
            fast=fast.next;
            count++;
        }
        while(fast!=null){
            fast=fast.next;
            slow=slow.next;
        }
        if(count<k) return null;
        return slow;
    }

```












<h3>13.反转链表：输入一个链表，反转链表后，输出新链表的表头。
</h3>

```
public class Solution {
    public ListNode ReverseList(ListNode head) {
        ListNode pre =null;
        ListNode cur = head;
        while(cur!=null){
            ListNode next= cur.next;
            cur.next=pre;
            pre=cur;
            cur=next;
    }
        return pre;
}}

```



















<h3>14.合并链表：输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
。
</h3>

```
方法1：

public class Solution {
    public ListNode Merge(ListNode list1,ListNode list2) {
        ListNode list= new ListNode(5);
        ListNode l =list;
        while (list1!=null&&list2!=null){
            if(list1.val<list2.val){
                    l.next=list1;
                    list1=list1.next;
            }
            else if(list1.val>=list2.val){
                    l.next=list2;
                     list2=list2.next;

            }
            l=l.next;
            
        }
        if(list1!=null) l.next=list1;
        if(list2!=null) l.next=list2;
        
        return list.next;

        
            

}
}

方法2：
public ListNode mergeTwoLists1(ListNode l1, ListNode l2) {
	        if(l1 == null) return l2;
	        if(l2 == null) return l1;
	        if(l1.val < l2.val){
	            l1.next = mergeTwoLists(l1.next, l2);
	            return l1;
	        } else{
	            l2.next = mergeTwoLists(l2.next, l1);
	            return l2;
	        }

	    }
```




















<h3>15.树的子结构：
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
。
</h3>

```
public class Solution {
    public static boolean HasSubtree(TreeNode root1, TreeNode root2) {
        boolean result = false;
        //当Tree1和Tree2都不为零的时候，才进行比较。否则直接返回false
        if (root2 != null && root1 != null) {
            //如果找到了对应Tree2的根节点的点
            if(root1.val == root2.val){
                //以这个根节点为为起点判断是否包含Tree2
                result = doesTree1HaveTree2(root1,root2);
            }
            //如果找不到，那么就再去root的左儿子当作起点，去判断时候包含Tree2
            if (!result) {
                result = HasSubtree(root1.left,root2);
            }
             
            //如果还找不到，那么就再去root的右儿子当作起点，去判断时候包含Tree2
            if (!result) {
                result = HasSubtree(root1.right,root2);
               }
            }
            //返回结果
        return result;
    }
 
    public static boolean doesTree1HaveTree2(TreeNode node1, TreeNode node2) {
        //如果Tree2已经遍历完了都能对应的上，返回true
        if (node2 == null) {
            return true;
        }
        //如果Tree2还没有遍历完，Tree1却遍历完了。返回false
        if (node1 == null) {
            return false;
        }
        //如果其中有一个点没有对应上，返回false
        if (node1.val != node2.val) {  
                return false;
        }
         
        //如果根节点对应的上，那么就分别去子节点里面匹配
        return doesTree1HaveTree2(node1.left,node2.left) && doesTree1HaveTree2(node1.right,node2.right);
    }
```





<h3>16.树的镜像：
操作给定的二叉树，将其变换为源二叉树的镜像。
。
</h3>

```
public void Mirror(TreeNode root) {
        if (root==null) return;
    	TreeNode temp =root.left;
    	root.left=root.right;
    	root.right=temp;
    	Mirror(root.left);
        Mirror(root.right);
    	   
       
        
        
    }

```











<h3>17.顺时针打印矩阵：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
。
</h3>

介绍一种矩阵处理方式：矩阵分圈处理。在矩阵中用左上角的坐标（tR，tC）和右下角的坐标（dR，dC）可以表示一个子矩阵，如题目中矩阵，当(tR,tC)=(0,0)、(dR,dC)=(3,3)时，表示的子矩阵就是整个矩阵，那么这个子矩阵的最外层的部分为：

1     2    3    4

5                8

9               12

13  14  15  16

把这个子矩阵的最外层顺时针打印出来，那么在(tR,tC)=(0,0)、(dR,dC)=(3,3)时，打印的结果为：1,2,3,4,8,12,16,15,14,13,9,5。接下来，分别使tR和tC加1，dR和dC减1，即(tR,tC)=(1,1)、(dR,dC)=(2,2)时，此时的子矩阵为：

 6     7    

10   11

再把这个矩阵顺时针打印出来，结果为6,7,11,10。再把tR和tC加1，dR和dC减1，即(tR,tC)=(2,2)、(dR,dC)=(1,1)。如果左上角坐标位于右下角坐标的右方或者下方（即tR>dR ||tC>dC），则停止，已经打印的所有结果即为要求的打印结果。



```
import java.util.ArrayList;
public class Solution {
    	 public ArrayList<Integer> printMatrix(int [][] matrix) {
		 ArrayList<Integer> list =new ArrayList<Integer>();
             if(matrix==null||matrix.length==0) 
		 {
			 return list;
		 }

		    int rowStart=0;
		    int colStart=0;
		    int rowEnd = matrix.length-1;
		    int colEnd = matrix[0].length-1;
		    while(rowStart<=rowEnd&&colStart<=colEnd) {
		    	helper(list,matrix,rowStart,rowEnd,colStart,colEnd);
		    	rowStart++;
		    	rowEnd--;
		    	colStart++;
		    	colEnd--;
		    }
		    return list;
	    }
	 public void helper(ArrayList<Integer> list,int[][] matrix ,
			 int rowStart,int rowEnd ,int colStart,int colEnd) {
		 if(rowStart==rowEnd) {
			 for(int i =colStart;i<=colEnd;i++) {
				 list.add(matrix[rowStart][i]);
			 }
		 }
		 else if(colStart==colEnd) {
			 for(int i =rowStart;i<=rowEnd;i++) {
				 list.add(matrix[i][colStart]);
			 }
		 }
		 else 
		 {
			 int curR=rowStart;
			 int curC=colStart;
			 while(curC!=colEnd) {
				 list.add(matrix[rowStart][curC]);
				 curC++;	 
			 }
while(curR!=rowEnd) { 
				 list.add(matrix[curR][colEnd]);
				 curR++;	 
			 }
while(curC!=colStart) {
	 
	 list.add(matrix[rowEnd][curC]);
	 curC--;	 
}
while(curR!=rowStart) {
	 
	 list.add(matrix[curR][colStart]);
	 curR--;	 
}			 
		 }
		 
	 }
}


```





























<h3>18.包含min函数的栈：
定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
</h3>






```


import java.util.Stack;

public class Solution {
	Stack<Integer> stack1 =new Stack<Integer>();
	Stack<Integer> stack2 =new Stack<Integer>();
    
    public void push(int node) {
	        stack1.push(node);
	        if(stack2.isEmpty()) {
	        	stack2.push(node);
	        	
	        }
	        else {
	        	int temp =stack2.peek();
	        	stack2.push(temp>node?node:temp);
	        }
	    }
	    
	    public void pop() {
	        stack1.pop();
	        stack2.pop();
	        
	    }
	    
	    public int top() {
	        return stack1.peek();
	    }
	    
	    public int min() {
	        return stack2.peek();
	    }

}
```


<h3>19.栈的压入，弹出序列：

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

</h3>

【思路】借用一个辅助的栈，遍历压栈顺序，先讲第一个放入栈中，这里是1，然后判断栈顶元素是不是出栈顺序的第一个元素，这里是4，很显然1≠4，所以我们继续压栈，直到相等以后开始出栈，出栈一个元素，则将出栈顺序向后移动一位，直到不相等，这样循环等压栈顺序遍历完成，如果辅助栈还不为空，说明弹出序列不是该栈的弹出顺序。

举例：

入栈1,2,3,4,5

出栈4,5,3,2,1

首先1入辅助栈，此时栈顶1≠4，继续入栈2

此时栈顶2≠4，继续入栈3

此时栈顶3≠4，继续入栈4

此时栈顶4＝4，出栈4，弹出序列向后一位，此时为5，,辅助栈里面是1,2,3

此时栈顶3≠5，继续入栈5

此时栈顶5=5，出栈5,弹出序列向后一位，此时为3，,辅助栈里面是1,2,3

….

依次执行，最后辅助栈为空。如果不为空说明弹出序列不是该栈的弹出顺序。
```


import java.util.ArrayList;
import java.util.Stack;
public class Solution {
    public boolean IsPopOrder(int [] pushA,int [] popA) {
        if(pushA.length == 0 || popA.length == 0)
            return false;
        Stack<Integer> s = new Stack<Integer>();
        //用于标识弹出序列的位置
        int popIndex = 0;
        for(int i = 0; i< pushA.length;i++){
            s.push(pushA[i]);
            //如果栈不为空，且栈顶元素等于弹出序列
            while(!s.empty() &&s.peek() == popA[popIndex]){
                //出栈
                s.pop();
                //弹出序列向后一位
                popIndex++;
            }
        }
        return s.empty();
    }
}
```

















<h3>20.从上往下打印出二叉树：
从上往下打印出二叉树的每个节点，同层节点从左至右打印。

</h3>


```
public class Solution {
   public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
	ArrayList<Integer> list =new ArrayList<Integer>();
	if (root==null) return list;
	Queue<TreeNode> queue = new LinkedList<TreeNode>();
     queue.offer(root);
     while(!queue.isEmpty()) {
    	 TreeNode temp = queue.poll();
    	 list.add(temp.val);
    	 if(temp.left!=null) {
    	 queue.offer(temp.left);}
    	 if(temp.right!=null) {
    	 queue.offer(temp.right);}
     }
     return list;
     
    }
}


```






























<h3>21.二叉搜索树的后序遍历序列:
	输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

</h3>


```


BST的后序序列的合法序列是，对于一个序列S，最后一个元素是x （也就是根），如果去掉
最后一个元素的序列为T，那么T满足：T可以分成两段，前一段（左子树）小于x，后一段（右
子树）大于x，且这两段（子树）都是合法的后序序列。完美的递归定义 : ) 。

public class Solution {

     public boolean VerifySquenceOfBST(int [] sequence) {
	        if(sequence.length==0) return false;
	        if(sequence.length==1) return true;
	        return helper(0,sequence,sequence.length-1);
	    }
	
	   public boolean helper(int start,int[] sequence,int root) {
		   if(start>=root ) {
			   return true;
		   }
		   int i = root;
		   while(i>start&&sequence[i-1]>sequence[root]) {
			   i--;
		   }
		   for(int j=start;j<i-1;j++) {
			   if(sequence[j]>sequence[root]) return false;
		   }
		   return helper(start,sequence,i-1)&&
				   helper(i,sequence,root-1);
	   }
}

```





















<h3>22.二叉树中和为某一值的路径
	
输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)
</h3>


```


public class Solution {
    ArrayList<ArrayList<Integer>> res = new ArrayList<>();
    ArrayList<Integer> path = new ArrayList<>();
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
        if (root == null) {
            return res;
        }
        findPath(root, target);
        return res;
    }
     
    public void findPath(TreeNode root, int target) {
        //因为FindPath中和 下面程序中都进行了判null操作，root绝对不可能为 null
        path.add(root.val);
        //已经到达叶子节点，并且正好加出了target
        if (root.val == target && root.left == null && root.right == null) {
            //将该路径加入res结果集中
            res.add(new ArrayList(path));
        }
        //如果左子树非空，递归左子树
        if (root.left != null) {
            findPath(root.left, target - root.val);
        }
        //如果右子树非空，递归右子树
        if (root.right != null) {
            findPath(root.right, target - root.val);
        }
        //无论当前路径是否加出了target，必须去掉最后一个，然后返回父节点，去查找另一条路径，最终的path肯定为null
        path.remove(path.size() - 1);
        return;
    }
     
}
```











<h3>23.复杂链表的复制

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
</h3>


```
/*
*解题思路：
*1、遍历链表，复制每个结点，如复制结点A得到A1，将结点A1插到结点A后面；
*2、重新遍历链表，复制老结点的随机指针给新结点，如A1.random = A.random.next;
*3、拆分链表，将链表拆分为原链表和复制后的链表
*/
public class Solution {
    public RandomListNode Clone(RandomListNode pHead) {
        if(pHead == null) {
            return null;
        }
         
        RandomListNode currentNode = pHead;
        //1、复制每个结点，如复制结点A得到A1，将结点A1插到结点A后面；
        while(currentNode != null){
            RandomListNode cloneNode = new RandomListNode(currentNode.label);
            RandomListNode nextNode = currentNode.next;
            currentNode.next = cloneNode;
            cloneNode.next = nextNode;
            currentNode = nextNode;
        }
         
        currentNode = pHead;
        //2、重新遍历链表，复制老结点的随机指针给新结点，如A1.random = A.random.next;
        while(currentNode != null) {
            currentNode.next.random = currentNode.random==null?null:currentNode.random.next;
            currentNode = currentNode.next.next;
        }
         
        //3、拆分链表，将链表拆分为原链表和复制后的链表
        currentNode = pHead;
        RandomListNode pCloneHead = pHead.next;
        while(currentNode != null) {
            RandomListNode cloneNode = currentNode.next;
            currentNode.next = cloneNode.next;
            cloneNode.next = cloneNode.next==null?null:cloneNode.next.next;
            currentNode = currentNode.next;
        }
         
        return pCloneHead;
    }
}

```




<h3>24.二叉搜索树转换成一个排序的双向链表

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
</h3>

```


解题思路：
1.将左子树构造成双链表，并返回链表头节点。
2.定位至左子树双链表最后一个节点。
3.如果左子树链表不为空的话，将当前root追加到左子树链表。
4.将右子树构造成双链表，并返回链表头节点。
5.如果右子树链表不为空的话，将该链表追加到root节点之后。
6.根据左子树链表是否为空确定返回的节点。
public class Solution {
     

    public TreeNode Convert(TreeNode root) {
      if(root==null)
            return null;
        if(root.left==null&&root.right==null)
            return root;
        // 1.将左子树构造成双链表，并返回链表头节点
        TreeNode left = Convert(root.left);
        TreeNode p = left;
        // 2.定位至左子树双链表最后一个节点
        while(p!=null&&p.right!=null){
            p = p.right;
        }
        // 3.如果左子树链表不为空的话，将当前root追加到左子树链表
        if(left!=null){
            p.right = root;
            root.left = p;
        }
        // 4.将右子树构造成双链表，并返回链表头节点
        TreeNode right = Convert(root.right);
        // 5.如果右子树链表不为空的话，将该链表追加到root节点之后
        if(right!=null){
            right.left = root;
            root.right = right;
        }
        return left!=null?left:root;  
    }
    
      
}

```




<h3>25.字符串的排列

输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则
打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
</h3>
```


public class Solution {
    public static void main(String[] args) {
        Solution p = new Solution();
        System.out.println(p.Permutation("abc").toString());
    }
 
    public ArrayList<String> Permutation(String str) {
        List<String> res = new ArrayList<>();
        if (str != null && str.length() > 0) {
            PermutationHelper(str.toCharArray(), 0, res);
            Collections.sort(res);
        }
        return (ArrayList)res;
    }
 
    public void PermutationHelper(char[] cs, int i, List<String> list) {
        if (i == cs.length - 1) {
            String val = String.valueOf(cs);
            if (!list.contains(val))
                list.add(val);
        } else {
            for (int j = i; j < cs.length; j++) {
                swap(cs, i, j);
                PermutationHelper(cs, i+1, list);
                swap(cs, i, j);
            }
        }
    }
 
    public void swap(char[] cs, int i, int j) {
        char temp = cs[i];
        cs[i] = cs[j];
        cs[j] = temp;
    }
}

```






















<h3>26.数组中出现次数超过一半的数字
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
	例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，
	超过数组长度的一半，因此输出2。如果不存在则输出0。
</h3>
```


```
解法一：

首先第一个for循环结束后得到的num是什么？如果这个数组中存在个数大于数组长度一半的数，
那么这个num一定是这个数，因为数组中所有不是num的数，一定会被这个数覆盖，所以最后得到的数是num。
但是，如果这个数组中根本不存在个数大于数组长度一半的数，那么这个num就是一个不确定的值，这也是为什么找出num之后，
还要再做一次循环验证这个数出现的个数是不是大于数组长度一半的原因。
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        int n = numbers.size();
        if (n == 0) return 0;
         
        int num = numbers[0], count = 1;
        for (int i = 1; i < n; i++) {
            if (numbers[i] == num) count++;
            else count--;
            if (count == 0) {
                num = numbers[i];
                count = 1;
            }
        }
        // Verifying
        count = 0;
        for (int i = 0; i < n; i++) {
            if (numbers[i] == num) count++;
        }
        if (count * 2 > n) return num;
        return 0;
    }
};



解法二：


import java.util.ArrayList;
import java.util.HashMap;
public class Solution {
   public static int MoreThanHalfNum_Solution(int [] array) {
	      if(array==null ) return 0;
         if(array.length==1)return 1;
	      int max=0;
	      HashMap<Integer,Integer> hs = new HashMap<Integer,Integer>();
	      for(int i = 0 ;i<array.length;i++) {
	    	  if(hs.containsKey(array[i])) { 
	    		  hs.put(array[i],hs.get(array[i])+1);
	    		  if(hs.get(array[i])>max) {
	    			  max=hs.get(array[i]);
//	    			  System.out.println(max);
	    		  }
	    		  
	    	  }
	    	  else {
	    		  hs.put(array[i], 1);
	    	  }
	      }
	      if(max>array.length/2) return 2;
	      return 0;
	    }
}

```








<h3>27.最小的k个数：输入n个整数，
找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

</h3>

```

用最大堆保存这k个数，每次只和堆顶比，如果比堆顶小，删除堆顶，新数入堆。

import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Comparator;
public class Solution {
   public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
       ArrayList<Integer> result = new ArrayList<Integer>();
       int length = input.length;
       if(k > length || k == 0){
           return result;
       }
        PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(k, new Comparator<Integer>() {
 
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }
        });
        for (int i = 0; i < length; i++) {
            if (maxHeap.size() != k) {
                maxHeap.offer(input[i]);
            } else if (maxHeap.peek() > input[i]) {
                Integer temp = maxHeap.poll();
                temp = null;
                maxHeap.offer(input[i]);
            }
        }
        for (Integer integer : maxHeap) {
            result.add(integer);
        }
        return result;
    }
```





<h3>28.连续子数组的最大和：
例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最
大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)
</h3>


```
public class Solution {
    public int FindGreatestSumOfSubArray(int[] array) {
		    int max=0x80000000;
		    int temp=0;
	        if(array.length==0) return 0;
	        for(int i =0;i<array.length;i++) {
	        	if(temp<=0) {
	        		temp =array[i];
	        	}
	        	else temp+=array[i];
	        	if(temp>max) {
	        		max=temp; 
	        	}
	        }
	        return max;
	        
	    }
}
```




<h3>29.整数中1出现的次数：
求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的
数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加
普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。


```
public class Solution {
      public static int NumberOf1Between1AndN_Solution(int n) {
		    
		 	if(n==0) return 0;
		 	StringBuffer s=new StringBuffer();		 	int count=0;
		 	for(int i =1;i<n+1;i++) {
		 		s.append(i);
		 		
		 	}
		 	 String str=s.toString();
//		 	char[] c=s.toCharArray();
////		 	System.out.println(c);
		 	for(int i=0;i<str.length();i++) {
		 		
		 		if(str.charAt(i)=='1') {
		 			
		 			count++;
		 		}
		 	}
		 	return count;
	    }
}
```



















<h3>30: 把数组排成最小的数字，输入一个正整数数组，把数组里所有数字拼接起来排成一个数，
打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。</h3>


```
public class Solution {
    public String PrintMinNumber(int [] numbers) {
        int n;
  String s="";
  ArrayList<Integer> list= new ArrayList<Integer>();
  n=numbers.length;
  for(int i=0;i<n;i++){
   list.add(numbers[i]);
   
  }
  Collections.sort(list, new Comparator<Integer>(){
  
  public int compare(Integer str1,Integer str2){
   String s1=str1+""+str2;
   String s2=str2+""+str1;
         return s1.compareTo(s2);
     }
  }   );
  
  for(int j:list){
   s+=j;
  }
        return s;

    }
}
```




<h3>31.丑数
	
把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
</h3>

```
该思路： 我们只用比较3个数：用于乘2的最小的数、用于乘3的最小的数，用于乘5的最小的
public class Solution {
    public int GetUglyNumber_Solution(int index) {
        if(index==0) return 0;
		ArrayList<Integer> list =new ArrayList<Integer>();
		list.add(1);
		int t2=0,t3=0,t5=0;
		for(int i =1;i<index;i++) {
			list.add(Math.min(list.get(t2)*2, Math.min(list.get(t3)*3, list.get(t5)*5 )));
			if(list.get(i)==list.get(t2)*2) t2++;
			if(list.get(i)==list.get(t3)*3) t3++;

			if(list.get(i)==list.get(t5)*5) t5++;

		}
		return list.get(list.size()-1);
		
    }
}
```















<h3>0：反转数字</h3>

举例：
Input: 123
Output: 321
```
public int reverse(int x) {
        long rev = 0; 
        while (x != 0) {
            rev = (rev * 10) + (x % 10);
            x /= 10;
        }
        if (rev > Integer.MAX_VALUE || rev < Integer.MIN_VALUE) return 0;
        return (int) rev;
    }

```
