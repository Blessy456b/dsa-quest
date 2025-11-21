"""
Striver's A2Z DSA Sheet Data Structure
Contains 450+ problems organized by 18 steps/topics
"""

STRIVER_SHEET = {
    "Step 1": {
        "name": "Learn the Basics",
        "description": "Master fundamental programming concepts and basic data structures",
        "icon": "🎯",
        "color": "#FF6B6B",
        "xp": 50,
        "problems": [
            {
                "id": 1,
                "title": "User Input/Output",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2",
                "concepts": ["Input/Output", "Basic Syntax"],
                "cpp_code": """#include <iostream>
using namespace std;

int main() {
    int n;
    cout << "Enter a number: ";
    cin >> n;
    cout << "You entered: " << n << endl;
    return 0;
}""",
                "java_code": """import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a number: ");
        int n = sc.nextInt();
        System.out.println("You entered: " + n);
        sc.close();
    }
}""",
                "python_code": """n = int(input("Enter a number: "))
print(f"You entered: {n}")"""
            },
            {
                "id": 2,
                "title": "Data Types",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2",
                "concepts": ["Variables", "Data Types"],
                "cpp_code": """#include <iostream>
using namespace std;

int main() {
    int a = 10;
    float b = 3.14;
    char c = 'A';
    string s = "Hello";
    cout << a << " " << b << " " << c << " " << s;
    return 0;
}""",
                "java_code": """public class Main {
    public static void main(String[] args) {
        int a = 10;
        float b = 3.14f;
        char c = 'A';
        String s = "Hello";
        System.out.println(a + " " + b + " " + c + " " + s);
    }
}""",
                "python_code": """a = 10
b = 3.14
c = 'A'
s = "Hello"
print(a, b, c, s)"""
            },
            {
                "id": 3,
                "title": "If-Else Statements",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2",
                "concepts": ["Conditionals", "Control Flow"],
                "cpp_code": """#include <iostream>
using namespace std;

int main() {
    int age = 18;
    if (age >= 18) {
        cout << "Adult" << endl;
    } else {
        cout << "Minor" << endl;
    }
    return 0;
}""",
                "java_code": """public class Main {
    public static void main(String[] args) {
        int age = 18;
        if (age >= 18) {
            System.out.println("Adult");
        } else {
            System.out.println("Minor");
        }
    }
}""",
                "python_code": """age = 18
if age >= 18:
    print("Adult")
else:
    print("Minor")"""
            }
        ]
    },
    "Step 2": {
        "name": "Important Sorting Techniques",
        "description": "Learn essential sorting algorithms and their applications",
        "icon": "🔄",
        "color": "#4ECDC4",
        "xp": 75,
        "problems": [
            {
                "id": 4,
                "title": "Selection Sort",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/selection-sort-algorithm",
                "concepts": ["Sorting", "Selection Sort"],
                "cpp_code": """#include <iostream>
#include <vector>
using namespace std;

void selectionSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; i++) {
        int minIdx = i;
        for (int j = i+1; j < n; j++) {
            if (arr[j] < arr[minIdx])
                minIdx = j;
        }
        swap(arr[i], arr[minIdx]);
    }
}""",
                "java_code": """public class Main {
    public static void selectionSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n-1; i++) {
            int minIdx = i;
            for (int j = i+1; j < n; j++) {
                if (arr[j] < arr[minIdx])
                    minIdx = j;
            }
            int temp = arr[i];
            arr[i] = arr[minIdx];
            arr[minIdx] = temp;
        }
    }
}""",
                "python_code": """def selection_sort(arr):
    n = len(arr)
    for i in range(n-1):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]"""
            },
            {
                "id": 5,
                "title": "Bubble Sort",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/bubble-sort-algorithm",
                "concepts": ["Sorting", "Bubble Sort"],
                "cpp_code": """#include <iostream>
#include <vector>
using namespace std;

void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}""",
                "java_code": """public class Main {
    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n-1; i++) {
            for (int j = 0; j < n-i-1; j++) {
                if (arr[j] > arr[j+1]) {
                    int temp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                }
            }
        }
    }
}""",
                "python_code": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n-1):
        for j in range(n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]"""
            }
        ]
    },
    "Step 3": {
        "name": "Arrays (Easy → Hard)",
        "description": "Master array manipulation, searching, and optimization",
        "icon": "📊",
        "color": "#95E1D3",
        "xp": 100,
        "problems": [
            {
                "id": 6,
                "title": "Largest Element in Array",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/find-the-largest-element-in-an-array",
                "concepts": ["Arrays", "Linear Search"],
                "cpp_code": """#include <iostream>
#include <vector>
using namespace std;

int findLargest(vector<int>& arr) {
    int maxVal = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        maxVal = max(maxVal, arr[i]);
    }
    return maxVal;
}""",
                "java_code": """public class Main {
    public static int findLargest(int[] arr) {
        int maxVal = arr[0];
        for (int i = 1; i < arr.length; i++) {
            maxVal = Math.max(maxVal, arr[i]);
        }
        return maxVal;
    }
}""",
                "python_code": """def find_largest(arr):
    max_val = arr[0]
    for num in arr:
        max_val = max(max_val, num)
    return max_val"""
            },
            {
                "id": 7,
                "title": "Second Largest Element",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/find-second-smallest-and-second-largest-element-in-an-array",
                "concepts": ["Arrays", "Linear Traversal"],
                "cpp_code": """#include <iostream>
#include <vector>
#include <climits>
using namespace std;

int findSecondLargest(vector<int>& arr) {
    int largest = INT_MIN, secondLargest = INT_MIN;
    for (int num : arr) {
        if (num > largest) {
            secondLargest = largest;
            largest = num;
        } else if (num > secondLargest && num != largest) {
            secondLargest = num;
        }
    }
    return secondLargest;
}""",
                "java_code": """public class Main {
    public static int findSecondLargest(int[] arr) {
        int largest = Integer.MIN_VALUE;
        int secondLargest = Integer.MIN_VALUE;
        for (int num : arr) {
            if (num > largest) {
                secondLargest = largest;
                largest = num;
            } else if (num > secondLargest && num != largest) {
                secondLargest = num;
            }
        }
        return secondLargest;
    }
}""",
                "python_code": """def find_second_largest(arr):
    largest = float('-inf')
    second_largest = float('-inf')
    for num in arr:
        if num > largest:
            second_largest = largest
            largest = num
        elif num > second_largest and num != largest:
            second_largest = num
    return second_largest"""
            },
            {
                "id": 8,
                "title": "Kadane's Algorithm (Max Subarray Sum)",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/kadanes-algorithm-maximum-subarray-sum-in-an-array",
                "concepts": ["Dynamic Programming", "Arrays"],
                "cpp_code": """#include <iostream>
#include <vector>
#include <climits>
using namespace std;

int maxSubArray(vector<int>& arr) {
    int maxSum = INT_MIN, currentSum = 0;
    for (int num : arr) {
        currentSum += num;
        maxSum = max(maxSum, currentSum);
        if (currentSum < 0) currentSum = 0;
    }
    return maxSum;
}""",
                "java_code": """public class Main {
    public static int maxSubArray(int[] arr) {
        int maxSum = Integer.MIN_VALUE;
        int currentSum = 0;
        for (int num : arr) {
            currentSum += num;
            maxSum = Math.max(maxSum, currentSum);
            if (currentSum < 0) currentSum = 0;
        }
        return maxSum;
    }
}""",
                "python_code": """def max_subarray(arr):
    max_sum = float('-inf')
    current_sum = 0
    for num in arr:
        current_sum += num
        max_sum = max(max_sum, current_sum)
        if current_sum < 0:
            current_sum = 0
    return max_sum"""
            }
        ]
    },
    "Step 4": {
        "name": "Binary Search",
        "description": "Master the divide and conquer search technique",
        "icon": "🔍",
        "color": "#F38181",
        "xp": 90,
        "problems": [
            {
                "id": 9,
                "title": "Binary Search Implementation",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/binary-search-explained",
                "concepts": ["Binary Search", "Divide and Conquer"],
                "cpp_code": """#include <iostream>
#include <vector>
using namespace std;

int binarySearch(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}""",
                "java_code": """public class Main {
    public static int binarySearch(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) return mid;
            else if (arr[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }
}""",
                "python_code": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"""
            }
        ]
    },
    "Step 5": {
        "name": "Strings",
        "description": "Learn string manipulation and pattern matching",
        "icon": "🔤",
        "color": "#AA96DA",
        "xp": 80,
        "problems": [
            {
                "id": 10,
                "title": "Reverse a String",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/reverse-a-string",
                "concepts": ["Strings", "Two Pointers"],
                "cpp_code": """#include <iostream>
#include <algorithm>
using namespace std;

string reverseString(string s) {
    reverse(s.begin(), s.end());
    return s;
}""",
                "java_code": """public class Main {
    public static String reverseString(String s) {
        return new StringBuilder(s).reverse().toString();
    }
}""",
                "python_code": """def reverse_string(s):
    return s[::-1]"""
            }
        ]
    },
    "Step 6": {
        "name": "Linked Lists",
        "description": "Master linked list operations and manipulations",
        "icon": "🔗",
        "color": "#FCBAD3",
        "xp": 95,
        "problems": [
            {
                "id": 11,
                "title": "Reverse a Linked List",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/reverse-a-linked-list",
                "concepts": ["Linked List", "Pointers"],
                "cpp_code": """struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    while (curr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}""",
                "java_code": """class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; }
}

public class Main {
    public static ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }
}""",
                "python_code": """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev"""
            }
        ]
    },
    "Step 7": {
        "name": "Recursion",
        "description": "Understand recursive problem-solving techniques",
        "icon": "🔁",
        "color": "#FFFFD2",
        "xp": 85,
        "problems": [
            {
                "id": 12,
                "title": "Factorial using Recursion",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/factorial-of-a-number",
                "concepts": ["Recursion", "Math"],
                "cpp_code": """#include <iostream>
using namespace std;

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}""",
                "java_code": """public class Main {
    public static int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
}""",
                "python_code": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""
            }
        ]
    },
    "Step 8": {
        "name": "Bit Manipulation",
        "description": "Learn bitwise operations and optimization tricks",
        "icon": "⚡",
        "color": "#A8E6CF",
        "xp": 70,
        "problems": [
            {
                "id": 13,
                "title": "Count Set Bits",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/bit-magic/count-total-set-bits",
                "concepts": ["Bit Manipulation", "Counting"],
                "cpp_code": """#include <iostream>
using namespace std;

int countSetBits(int n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}""",
                "java_code": """public class Main {
    public static int countSetBits(int n) {
        int count = 0;
        while (n > 0) {
            count += n & 1;
            n >>= 1;
        }
        return count;
    }
}""",
                "python_code": """def count_set_bits(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count"""
            }
        ]
    },
    "Step 9": {
        "name": "Stack & Queues",
        "description": "Master LIFO and FIFO data structures",
        "icon": "📚",
        "color": "#FFD3B6",
        "xp": 88,
        "problems": [
            {
                "id": 14,
                "title": "Implement Stack using Array",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/implement-stack-using-array",
                "concepts": ["Stack", "Arrays"],
                "cpp_code": """#include <iostream>
#include <vector>
using namespace std;

class Stack {
    vector<int> arr;
public:
    void push(int x) { arr.push_back(x); }
    void pop() { if (!arr.empty()) arr.pop_back(); }
    int top() { return arr.empty() ? -1 : arr.back(); }
    bool isEmpty() { return arr.empty(); }
};""",
                "java_code": """import java.util.ArrayList;

class Stack {
    private ArrayList<Integer> arr = new ArrayList<>();
    
    public void push(int x) { arr.add(x); }
    public void pop() { if (!arr.isEmpty()) arr.remove(arr.size()-1); }
    public int top() { return arr.isEmpty() ? -1 : arr.get(arr.size()-1); }
    public boolean isEmpty() { return arr.isEmpty(); }
}""",
                "python_code": """class Stack:
    def __init__(self):
        self.arr = []
    
    def push(self, x):
        self.arr.append(x)
    
    def pop(self):
        if self.arr:
            self.arr.pop()
    
    def top(self):
        return self.arr[-1] if self.arr else -1
    
    def is_empty(self):
        return len(self.arr) == 0"""
            }
        ]
    },
    "Step 10": {
        "name": "Sliding Window & Two Pointer",
        "description": "Optimize array problems with window techniques",
        "icon": "🪟",
        "color": "#FFAAA5",
        "xp": 92,
        "problems": [
            {
                "id": 15,
                "title": "Max Sum Subarray of size K",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/sliding-window-technique",
                "concepts": ["Sliding Window", "Arrays"],
                "cpp_code": """#include <iostream>
#include <vector>
using namespace std;

int maxSumSubarray(vector<int>& arr, int k) {
    int maxSum = 0, windowSum = 0;
    for (int i = 0; i < k; i++) windowSum += arr[i];
    maxSum = windowSum;
    
    for (int i = k; i < arr.size(); i++) {
        windowSum += arr[i] - arr[i-k];
        maxSum = max(maxSum, windowSum);
    }
    return maxSum;
}""",
                "java_code": """public class Main {
    public static int maxSumSubarray(int[] arr, int k) {
        int maxSum = 0, windowSum = 0;
        for (int i = 0; i < k; i++) windowSum += arr[i];
        maxSum = windowSum;
        
        for (int i = k; i < arr.length; i++) {
            windowSum += arr[i] - arr[i-k];
            maxSum = Math.max(maxSum, windowSum);
        }
        return maxSum;
    }
}""",
                "python_code": """def max_sum_subarray(arr, k):
    max_sum = window_sum = sum(arr[:k])
    
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum"""
            }
        ]
    },
    "Step 11": {
        "name": "Heaps",
        "description": "Learn priority queue and heap operations",
        "icon": "🏔️",
        "color": "#FF6B9D",
        "xp": 93,
        "problems": [
            {
                "id": 16,
                "title": "Kth Largest Element",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/kth-largest-smallest-element-in-an-array",
                "concepts": ["Heap", "Priority Queue"],
                "cpp_code": """#include <queue>
#include <vector>
using namespace std;

int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;
    for (int num : nums) {
        minHeap.push(num);
        if (minHeap.size() > k) {
            minHeap.pop();
        }
    }
    return minHeap.top();
}""",
                "java_code": """import java.util.PriorityQueue;

public class Main {
    public static int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int num : nums) {
            minHeap.offer(num);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }
        return minHeap.peek();
    }
}""",
                "python_code": """import heapq

def find_kth_largest(nums, k):
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    return min_heap[0]"""
            },
            {
                "id": 17,
                "title": "Merge K Sorted Lists",
                "difficulty": "Hard",
                "link": "https://takeuforward.org/data-structure/merge-k-sorted-arrays",
                "concepts": ["Heap", "Linked List", "Merge"],
                "cpp_code": """#include <queue>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

struct Compare {
    bool operator()(ListNode* a, ListNode* b) {
        return a->val > b->val;
    }
};

ListNode* mergeKLists(vector<ListNode*>& lists) {
    priority_queue<ListNode*, vector<ListNode*>, Compare> pq;
    for (auto list : lists) {
        if (list) pq.push(list);
    }
    
    ListNode dummy(0);
    ListNode* curr = &dummy;
    
    while (!pq.empty()) {
        ListNode* node = pq.top();
        pq.pop();
        curr->next = node;
        curr = curr->next;
        if (node->next) pq.push(node->next);
    }
    
    return dummy.next;
}""",
                "java_code": """import java.util.PriorityQueue;

class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; }
}

public class Main {
    public static ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
        for (ListNode list : lists) {
            if (list != null) pq.offer(list);
        }
        
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;
        
        while (!pq.isEmpty()) {
            ListNode node = pq.poll();
            curr.next = node;
            curr = curr.next;
            if (node.next != null) pq.offer(node.next);
        }
        
        return dummy.next;
    }
}""",
                "python_code": """import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_lists(lists):
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    
    dummy = ListNode(0)
    curr = dummy
    counter = len(lists)
    
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, counter, node.next))
            counter += 1
    
    return dummy.next"""
            },
            {
                "id": 18,
                "title": "Top K Frequent Elements",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/top-k-frequent-elements",
                "concepts": ["Heap", "HashMap", "Frequency"],
                "cpp_code": """#include <vector>
#include <unordered_map>
#include <queue>
using namespace std;

vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (int num : nums) freq[num]++;
    
    auto comp = [](pair<int,int>& a, pair<int,int>& b) {
        return a.second > b.second;
    };
    priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(comp)> pq(comp);
    
    for (auto& p : freq) {
        pq.push(p);
        if (pq.size() > k) pq.pop();
    }
    
    vector<int> result;
    while (!pq.empty()) {
        result.push_back(pq.top().first);
        pq.pop();
    }
    return result;
}""",
                "java_code": """import java.util.*;

public class Main {
    public static int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int num : nums) {
            freq.put(num, freq.getOrDefault(num, 0) + 1);
        }
        
        PriorityQueue<Map.Entry<Integer, Integer>> pq = 
            new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
        
        for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
            pq.offer(entry);
            if (pq.size() > k) pq.poll();
        }
        
        int[] result = new int[k];
        for (int i = 0; i < k; i++) {
            result[i] = pq.poll().getKey();
        }
        return result;
    }
}""",
                "python_code": """from collections import Counter
import heapq

def top_k_frequent(nums, k):
    freq = Counter(nums)
    return heapq.nlargest(k, freq.keys(), key=freq.get)"""
            }
        ]
    },
    "Step 12": {
        "name": "Greedy Algorithms",
        "description": "Make locally optimal choices for global solutions",
        "icon": "🎁",
        "color": "#C1C8E4",
        "xp": 87,
        "problems": [
            {
                "id": 19,
                "title": "Activity Selection Problem",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/n-meetings-in-one-room",
                "concepts": ["Greedy", "Sorting"],
                "cpp_code": """#include <vector>
#include <algorithm>
using namespace std;

int activitySelection(vector<int>& start, vector<int>& end) {
    vector<pair<int, int>> activities;
    for (int i = 0; i < start.size(); i++) {
        activities.push_back({end[i], start[i]});
    }
    sort(activities.begin(), activities.end());
    
    int count = 1;
    int lastEnd = activities[0].first;
    
    for (int i = 1; i < activities.size(); i++) {
        if (activities[i].second >= lastEnd) {
            count++;
            lastEnd = activities[i].first;
        }
    }
    return count;
}""",
                "java_code": """import java.util.*;

public class Main {
    public static int activitySelection(int[] start, int[] end) {
        int n = start.length;
        int[][] activities = new int[n][2];
        for (int i = 0; i < n; i++) {
            activities[i][0] = end[i];
            activities[i][1] = start[i];
        }
        Arrays.sort(activities, (a, b) -> a[0] - b[0]);
        
        int count = 1;
        int lastEnd = activities[0][0];
        
        for (int i = 1; i < n; i++) {
            if (activities[i][1] >= lastEnd) {
                count++;
                lastEnd = activities[i][0];
            }
        }
        return count;
    }
}""",
                "python_code": """def activity_selection(start, end):
    activities = sorted(zip(end, start))
    count = 1
    last_end = activities[0][0]
    
    for i in range(1, len(activities)):
        if activities[i][1] >= last_end:
            count += 1
            last_end = activities[i][0]
    
    return count"""
            },
            {
                "id": 20,
                "title": "Fractional Knapsack",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/fractional-knapsack",
                "concepts": ["Greedy", "Sorting"],
                "cpp_code": """#include <vector>
#include <algorithm>
using namespace std;

double fractionalKnapsack(int W, vector<int>& values, vector<int>& weights) {
    vector<pair<double, pair<int,int>>> items;
    for (int i = 0; i < values.size(); i++) {
        double ratio = (double)values[i] / weights[i];
        items.push_back({ratio, {values[i], weights[i]}});
    }
    sort(items.rbegin(), items.rend());
    
    double maxValue = 0.0;
    for (auto& item : items) {
        if (W >= item.second.second) {
            maxValue += item.second.first;
            W -= item.second.second;
        } else {
            maxValue += item.first * W;
            break;
        }
    }
    return maxValue;
}""",
                "java_code": """import java.util.*;

public class Main {
    public static double fractionalKnapsack(int W, int[] values, int[] weights) {
        int n = values.length;
        double[][] items = new double[n][3];
        for (int i = 0; i < n; i++) {
            items[i][0] = (double)values[i] / weights[i];
            items[i][1] = values[i];
            items[i][2] = weights[i];
        }
        Arrays.sort(items, (a, b) -> Double.compare(b[0], a[0]));
        
        double maxValue = 0.0;
        for (double[] item : items) {
            if (W >= item[2]) {
                maxValue += item[1];
                W -= item[2];
            } else {
                maxValue += item[0] * W;
                break;
            }
        }
        return maxValue;
    }
}""",
                "python_code": """def fractional_knapsack(W, values, weights):
    items = [(v/w, v, w) for v, w in zip(values, weights)]
    items.sort(reverse=True)
    
    max_value = 0.0
    for ratio, value, weight in items:
        if W >= weight:
            max_value += value
            W -= weight
        else:
            max_value += ratio * W
            break
    
    return max_value"""
            }
        ]
    },
    "Step 13": {
        "name": "Binary Trees",
        "description": "Master tree traversals and operations",
        "icon": "🌳",
        "color": "#8EE4AF",
        "xp": 98,
        "problems": [
            {
                "id": 21,
                "title": "Binary Tree Inorder Traversal",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/binary-tree-traversals",
                "concepts": ["Binary Tree", "Recursion", "DFS"],
                "cpp_code": """struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

void inorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);
    result.push_back(root->val);
    inorder(root->right, result);
}

vector<int> inorderTraversal(TreeNode* root) {
    vector<int> result;
    inorder(root, result);
    return result;
}""",
                "java_code": """class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int x) { val = x; }
}

public class Main {
    public static List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        inorder(root, result);
        return result;
    }
    
    private static void inorder(TreeNode root, List<Integer> result) {
        if (root == null) return;
        inorder(root.left, result);
        result.add(root.val);
        inorder(root.right, result);
    }
}""",
                "python_code": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    result = []
    
    def inorder(node):
        if not node:
            return
        inorder(node.left)
        result.append(node.val)
        inorder(node.right)
    
    inorder(root)
    return result"""
            },
            {
                "id": 22,
                "title": "Maximum Depth of Binary Tree",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/maximum-depth-of-a-binary-tree",
                "concepts": ["Binary Tree", "Recursion", "DFS"],
                "cpp_code": """struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

int maxDepth(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}""",
                "java_code": """class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int x) { val = x; }
}

public class Main {
    public static int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }
}""",
                "python_code": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))"""
            },
            {
                "id": 23,
                "title": "Diameter of Binary Tree",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/calculate-the-diameter-of-a-binary-tree",
                "concepts": ["Binary Tree", "Recursion"],
                "cpp_code": """struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

int diameter = 0;

int height(TreeNode* root) {
    if (!root) return 0;
    int left = height(root->left);
    int right = height(root->right);
    diameter = max(diameter, left + right);
    return 1 + max(left, right);
}

int diameterOfBinaryTree(TreeNode* root) {
    diameter = 0;
    height(root);
    return diameter;
}""",
                "java_code": """class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int x) { val = x; }
}

public class Main {
    static int diameter = 0;
    
    public static int diameterOfBinaryTree(TreeNode root) {
        diameter = 0;
        height(root);
        return diameter;
    }
    
    private static int height(TreeNode root) {
        if (root == null) return 0;
        int left = height(root.left);
        int right = height(root.right);
        diameter = Math.max(diameter, left + right);
        return 1 + Math.max(left, right);
    }
}""",
                "python_code": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def diameter_of_binary_tree(root):
    diameter = 0
    
    def height(node):
        nonlocal diameter
        if not node:
            return 0
        left = height(node.left)
        right = height(node.right)
        diameter = max(diameter, left + right)
        return 1 + max(left, right)
    
    height(root)
    return diameter"""
            }
        ]
    },
    "Step 14": {
        "name": "Binary Search Trees",
        "description": "Learn ordered tree operations",
        "icon": "🌲",
        "color": "#EDF5E1",
        "xp": 94,
        "problems": [
            {
                "id": 24,
                "title": "Search in BST",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/search-in-a-binary-search-tree",
                "concepts": ["BST", "Binary Search"],
                "cpp_code": """struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* searchBST(TreeNode* root, int val) {
    if (!root || root->val == val) return root;
    if (val < root->val) return searchBST(root->left, val);
    return searchBST(root->right, val);
}""",
                "java_code": """class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int x) { val = x; }
}

public class Main {
    public static TreeNode searchBST(TreeNode root, int val) {
        if (root == null || root.val == val) return root;
        if (val < root.val) return searchBST(root.left, val);
        return searchBST(root.right, val);
    }
}""",
                "python_code": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def search_bst(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return search_bst(root.left, val)
    return search_bst(root.right, val)"""
            },
            {
                "id": 25,
                "title": "Validate BST",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/check-if-a-tree-is-a-binary-search-tree-or-binary-tree",
                "concepts": ["BST", "Recursion"],
                "cpp_code": """struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

bool isValidBST(TreeNode* root, long min = LONG_MIN, long max = LONG_MAX) {
    if (!root) return true;
    if (root->val <= min || root->val >= max) return false;
    return isValidBST(root->left, min, root->val) && 
           isValidBST(root->right, root->val, max);
}""",
                "java_code": """class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int x) { val = x; }
}

public class Main {
    public static boolean isValidBST(TreeNode root) {
        return isValid(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }
    
    private static boolean isValid(TreeNode root, long min, long max) {
        if (root == null) return true;
        if (root.val <= min || root.val >= max) return false;
        return isValid(root.left, min, root.val) && 
               isValid(root.right, root.val, max);
    }
}""",
                "python_code": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (is_valid_bst(root.left, min_val, root.val) and
            is_valid_bst(root.right, root.val, max_val))"""
            }
        ]
    },
    "Step 15": {
        "name": "Graphs",
        "description": "Explore graph algorithms and traversals",
        "icon": "🕸️",
        "color": "#05386B",
        "xp": 110,
        "problems": [
            {
                "id": 26,
                "title": "BFS Traversal",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/graph/breadth-first-search-bfs",
                "concepts": ["Graph", "BFS", "Queue"],
                "cpp_code": """#include <vector>
#include <queue>
using namespace std;

vector<int> bfsTraversal(int V, vector<int> adj[]) {
    vector<int> result;
    vector<bool> visited(V, false);
    queue<int> q;
    
    q.push(0);
    visited[0] = true;
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);
        
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
    return result;
}""",
                "java_code": """import java.util.*;

public class Main {
    public static List<Integer> bfsTraversal(int V, List<List<Integer>> adj) {
        List<Integer> result = new ArrayList<>();
        boolean[] visited = new boolean[V];
        Queue<Integer> q = new LinkedList<>();
        
        q.add(0);
        visited[0] = true;
        
        while (!q.isEmpty()) {
            int node = q.poll();
            result.add(node);
            
            for (int neighbor : adj.get(node)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.add(neighbor);
                }
            }
        }
        return result;
    }
}""",
                "python_code": """from collections import deque

def bfs_traversal(V, adj):
    result = []
    visited = [False] * V
    q = deque([0])
    visited[0] = True
    
    while q:
        node = q.popleft()
        result.append(node)
        
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                q.append(neighbor)
    
    return result"""
            },
            {
                "id": 27,
                "title": "DFS Traversal",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/depth-first-search-dfs",
                "concepts": ["Graph", "DFS", "Recursion"],
                "cpp_code": """#include <vector>
using namespace std;

void dfs(int node, vector<int> adj[], vector<bool>& visited, vector<int>& result) {
    visited[node] = true;
    result.push_back(node);
    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            dfs(neighbor, adj, visited, result);
        }
    }
}

vector<int> dfsTraversal(int V, vector<int> adj[]) {
    vector<int> result;
    vector<bool> visited(V, false);
    dfs(0, adj, visited, result);
    return result;
}""",
                "java_code": """import java.util.*;

public class Main {
    public static List<Integer> dfsTraversal(int V, List<List<Integer>> adj) {
        List<Integer> result = new ArrayList<>();
        boolean[] visited = new boolean[V];
        dfs(0, adj, visited, result);
        return result;
    }
    
    private static void dfs(int node, List<List<Integer>> adj, 
                           boolean[] visited, List<Integer> result) {
        visited[node] = true;
        result.add(node);
        for (int neighbor : adj.get(node)) {
            if (!visited[neighbor]) {
                dfs(neighbor, adj, visited, result);
            }
        }
    }
}""",
                "python_code": """def dfs_traversal(V, adj):
    result = []
    visited = [False] * V
    
    def dfs(node):
        visited[node] = True
        result.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor)
    
    dfs(0)
    return result"""
            },
            {
                "id": 28,
                "title": "Detect Cycle in Undirected Graph",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/detect-cycle-in-an-undirected-graph-using-bfs",
                "concepts": ["Graph", "Cycle Detection", "DFS"],
                "cpp_code": """#include <vector>
using namespace std;

bool hasCycleDFS(int node, int parent, vector<int> adj[], vector<bool>& visited) {
    visited[node] = true;
    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            if (hasCycleDFS(neighbor, node, adj, visited))
                return true;
        } else if (neighbor != parent) {
            return true;
        }
    }
    return false;
}

bool detectCycle(int V, vector<int> adj[]) {
    vector<bool> visited(V, false);
    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            if (hasCycleDFS(i, -1, adj, visited))
                return true;
        }
    }
    return false;
}""",
                "java_code": """import java.util.*;

public class Main {
    public static boolean detectCycle(int V, List<List<Integer>> adj) {
        boolean[] visited = new boolean[V];
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                if (hasCycleDFS(i, -1, adj, visited))
                    return true;
            }
        }
        return false;
    }
    
    private static boolean hasCycleDFS(int node, int parent, 
                                      List<List<Integer>> adj, boolean[] visited) {
        visited[node] = true;
        for (int neighbor : adj.get(node)) {
            if (!visited[neighbor]) {
                if (hasCycleDFS(neighbor, node, adj, visited))
                    return true;
            } else if (neighbor != parent) {
                return true;
            }
        }
        return false;
    }
}""",
                "python_code": """def detect_cycle(V, adj):
    visited = [False] * V
    
    def has_cycle_dfs(node, parent):
        visited[node] = True
        for neighbor in adj[node]:
            if not visited[neighbor]:
                if has_cycle_dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False
    
    for i in range(V):
        if not visited[i]:
            if has_cycle_dfs(i, -1):
                return True
    return False"""
            }
        ]
    },
    "Step 16": {
        "name": "Dynamic Programming",
        "description": "Optimize with memoization and tabulation",
        "icon": "💎",
        "color": "#379683",
        "xp": 120,
        "problems": [
            {
                "id": 29,
                "title": "Fibonacci Number",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/dynamic-programming-introduction",
                "concepts": ["DP", "Memoization", "Tabulation"],
                "cpp_code": """#include <vector>
using namespace std;

int fibonacci(int n, vector<int>& dp) {
    if (n <= 1) return n;
    if (dp[n] != -1) return dp[n];
    return dp[n] = fibonacci(n-1, dp) + fibonacci(n-2, dp);
}

int fib(int n) {
    vector<int> dp(n+1, -1);
    return fibonacci(n, dp);
}""",
                "java_code": """public class Main {
    public static int fib(int n) {
        int[] dp = new int[n+1];
        for (int i = 0; i <= n; i++) dp[i] = -1;
        return fibonacci(n, dp);
    }
    
    private static int fibonacci(int n, int[] dp) {
        if (n <= 1) return n;
        if (dp[n] != -1) return dp[n];
        return dp[n] = fibonacci(n-1, dp) + fibonacci(n-2, dp);
    }
}""",
                "python_code": """def fib(n):
    dp = [-1] * (n + 1)
    
    def fibonacci(num):
        if num <= 1:
            return num
        if dp[num] != -1:
            return dp[num]
        dp[num] = fibonacci(num-1) + fibonacci(num-2)
        return dp[num]
    
    return fibonacci(n)"""
            },
            {
                "id": 30,
                "title": "Climbing Stairs",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/dynamic-programming-climbing-stairs",
                "concepts": ["DP", "Fibonacci", "Optimization"],
                "cpp_code": """int climbStairs(int n) {
    if (n <= 2) return n;
    int prev2 = 1, prev1 = 2;
    for (int i = 3; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}""",
                "java_code": """public class Main {
    public static int climbStairs(int n) {
        if (n <= 2) return n;
        int prev2 = 1, prev1 = 2;
        for (int i = 3; i <= n; i++) {
            int curr = prev1 + prev2;
            prev2 = prev1;
            prev1 = curr;
        }
        return prev1;
    }
}""",
                "python_code": """def climb_stairs(n):
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr
    return prev1"""
            },
            {
                "id": 31,
                "title": "Longest Increasing Subsequence",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/longest-increasing-subsequence",
                "concepts": ["DP", "LIS", "Binary Search"],
                "cpp_code": """#include <vector>
#include <algorithm>
using namespace std;

int lengthOfLIS(vector<int>& nums) {
    vector<int> dp;
    for (int num : nums) {
        auto it = lower_bound(dp.begin(), dp.end(), num);
        if (it == dp.end()) {
            dp.push_back(num);
        } else {
            *it = num;
        }
    }
    return dp.size();
}""",
                "java_code": """import java.util.*;

public class Main {
    public static int lengthOfLIS(int[] nums) {
        List<Integer> dp = new ArrayList<>();
        for (int num : nums) {
            int pos = Collections.binarySearch(dp, num);
            if (pos < 0) pos = -(pos + 1);
            if (pos == dp.size()) {
                dp.add(num);
            } else {
                dp.set(pos, num);
            }
        }
        return dp.size();
    }
}""",
                "python_code": """import bisect

def length_of_lis(nums):
    dp = []
    for num in nums:
        pos = bisect.bisect_left(dp, num)
        if pos == len(dp):
            dp.append(num)
        else:
            dp[pos] = num
    return len(dp)"""
            }
        ]
    },
    "Step 17": {
        "name": "Tries",
        "description": "Learn prefix tree data structure",
        "icon": "🔤",
        "color": "#5CDB95",
        "xp": 75,
        "problems": [
            {
                "id": 32,
                "title": "Implement Trie",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/implement-trie",
                "concepts": ["Trie", "Prefix Tree"],
                "cpp_code": """class TrieNode {
public:
    TrieNode* children[26];
    bool isEnd;
    TrieNode() {
        isEnd = false;
        for (int i = 0; i < 26; i++)
            children[i] = nullptr;
    }
};

class Trie {
    TrieNode* root;
public:
    Trie() { root = new TrieNode(); }
    
    void insert(string word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx])
                node->children[idx] = new TrieNode();
            node = node->children[idx];
        }
        node->isEnd = true;
    }
    
    bool search(string word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) return false;
            node = node->children[idx];
        }
        return node->isEnd;
    }
    
    bool startsWith(string prefix) {
        TrieNode* node = root;
        for (char c : prefix) {
            int idx = c - 'a';
            if (!node->children[idx]) return false;
            node = node->children[idx];
        }
        return true;
    }
};""",
                "java_code": """class TrieNode {
    TrieNode[] children = new TrieNode[26];
    boolean isEnd = false;
}

class Trie {
    TrieNode root;
    
    public Trie() {
        root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int idx = c - 'a';
            if (node.children[idx] == null)
                node.children[idx] = new TrieNode();
            node = node.children[idx];
        }
        node.isEnd = true;
    }
    
    public boolean search(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int idx = c - 'a';
            if (node.children[idx] == null) return false;
            node = node.children[idx];
        }
        return node.isEnd;
    }
    
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char c : prefix.toCharArray()) {
            int idx = c - 'a';
            if (node.children[idx] == null) return false;
            node = node.children[idx];
        }
        return true;
    }
}""",
                "python_code": """class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True"""
            }
        ]
    },
    "Step 18": {
        "name": "Strings (Advanced)",
        "description": "Master advanced string algorithms",
        "icon": "📝",
        "color": "#EDF5E1",
        "xp": 85,
        "problems": [
            {
                "id": 33,
                "title": "Longest Palindromic Substring",
                "difficulty": "Medium",
                "link": "https://takeuforward.org/data-structure/print-longest-palindromic-substring",
                "concepts": ["Strings", "DP", "Palindrome"],
                "cpp_code": """#include <string>
using namespace std;

string longestPalindrome(string s) {
    int n = s.length();
    if (n < 2) return s;
    
    int start = 0, maxLen = 1;
    
    auto expandAroundCenter = [&](int left, int right) {
        while (left >= 0 && right < n && s[left] == s[right]) {
            if (right - left + 1 > maxLen) {
                start = left;
                maxLen = right - left + 1;
            }
            left--;
            right++;
        }
    };
    
    for (int i = 0; i < n; i++) {
        expandAroundCenter(i, i);
        expandAroundCenter(i, i + 1);
    }
    
    return s.substr(start, maxLen);
}""",
                "java_code": """public class Main {
    public static String longestPalindrome(String s) {
        if (s.length() < 2) return s;
        int start = 0, maxLen = 1;
        
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > maxLen) {
                start = i - (len - 1) / 2;
                maxLen = len;
            }
        }
        
        return s.substring(start, start + maxLen);
    }
    
    private static int expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }
}""",
                "python_code": """def longest_palindrome(s):
    if len(s) < 2:
        return s
    
    start = 0
    max_len = 1
    
    def expand_around_center(left, right):
        nonlocal start, max_len
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right - left + 1 > max_len:
                start = left
                max_len = right - left + 1
            left -= 1
            right += 1
    
    for i in range(len(s)):
        expand_around_center(i, i)
        expand_around_center(i, i + 1)
    
    return s[start:start + max_len]"""
            },
            {
                "id": 34,
                "title": "Implement strStr()",
                "difficulty": "Easy",
                "link": "https://takeuforward.org/data-structure/kmp-algorithm-for-pattern-searching",
                "concepts": ["Strings", "Pattern Matching", "KMP"],
                "cpp_code": """#include <string>
#include <vector>
using namespace std;

int strStr(string haystack, string needle) {
    int m = haystack.length(), n = needle.length();
    if (n == 0) return 0;
    
    for (int i = 0; i <= m - n; i++) {
        int j;
        for (j = 0; j < n; j++) {
            if (haystack[i + j] != needle[j])
                break;
        }
        if (j == n) return i;
    }
    return -1;
}""",
                "java_code": """public class Main {
    public static int strStr(String haystack, String needle) {
        int m = haystack.length(), n = needle.length();
        if (n == 0) return 0;
        
        for (int i = 0; i <= m - n; i++) {
            int j;
            for (j = 0; j < n; j++) {
                if (haystack.charAt(i + j) != needle.charAt(j))
                    break;
            }
            if (j == n) return i;
        }
        return -1;
    }
}""",
                "python_code": """def str_str(haystack, needle):
    if not needle:
        return 0
    
    m, n = len(haystack), len(needle)
    
    for i in range(m - n + 1):
        if haystack[i:i+n] == needle:
            return i
    
    return -1"""
            }
        ]
    }
}

# Total problems count
TOTAL_PROBLEMS = sum(len(step["problems"]) for step in STRIVER_SHEET.values())

# Badge system
BADGES = {
    "rookie": {"name": "🌱 DSA Rookie", "xp_required": 0, "color": "#90EE90"},
    "explorer": {"name": "🎯 Code Explorer", "xp_required": 100, "color": "#87CEEB"},
    "warrior": {"name": "⚔️ Algorithm Warrior", "xp_required": 300, "color": "#FFD700"},
    "master": {"name": "🏆 DSA Master", "xp_required": 600, "color": "#FF6347"},
    "legend": {"name": "👑 Coding Legend", "xp_required": 1000, "color": "#8A2BE2"}
}
